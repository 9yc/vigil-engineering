"""Deterministic multi-turn conversation canonicalizer, window fitter, and byte-level BPE."""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union


VALID_ROLES = frozenset({"system", "user", "assistant", "tool"})
TRUNCATION_POLICIES = frozenset({"sliding_window", "middle_out"})
_PRETOKENIZE_RE = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]{0,127}$")
_SPECIAL_MARKERS = ("<|tools|>", "<|/tools|>", "<|msg|>", "<|/msg|>", "<|tool_calls|>", "<|/tool_calls|>")


class SerializationError(Exception):
    """Base error for conversation canonicalization, validation, and tokenization."""


class ToolSchemaError(SerializationError):
    """Raised when a tool definition or tool invocation is malformed."""


class ConversationValidationError(SerializationError):
    """Raised when role transitions or tool_call_id bindings are invalid."""


class BudgetExceededError(SerializationError):
    """Raised when pinned messages, tool schemas, and latest user turn exceed max_tokens."""


@functools.lru_cache(maxsize=1)
def bytes_to_unicode() -> Dict[int, str]:
    """Build a reversible bijection between byte values `0..255` and printable chars."""
    printable = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    chars, extra = list(printable), 0
    for bval in range(256):
        if bval not in printable:
            printable.append(bval)
            chars.append(256 + extra)
            extra += 1
    return {b: chr(c) for b, c in zip(printable, chars)}


@functools.lru_cache(maxsize=1)
def unicode_to_bytes() -> Dict[str, int]:
    return {ch: b for b, ch in bytes_to_unicode().items()}


def build_base_byte_vocab() -> Dict[str, int]:
    b2u = bytes_to_unicode()
    return {b2u[b]: b for b in range(256)}


class ByteLevelBPE:
    """Pure-Python byte-level BPE encoder/decoder with exact rank-priority merging."""

    def __init__(
        self, vocab: Mapping[str, int], merges: Sequence[Tuple[str, str]], *,
        special_tokens: Optional[Mapping[str, int]] = None, pretokenize: bool = True,
    ) -> None:
        if not isinstance(vocab, Mapping) or not vocab:
            raise SerializationError("vocab must be a non-empty mapping of token -> id")
        self._b2u, self._u2b = bytes_to_unicode(), unicode_to_bytes()
        self._vocab: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}

        for tok, idx in vocab.items():
            if not isinstance(tok, str) or not tok or isinstance(idx, bool) or not isinstance(idx, int):
                raise SerializationError(f"Invalid vocab entry: {tok!r} -> {idx!r}")
            if idx < 0 or idx in self._id_to_token:
                raise SerializationError(f"Negative or duplicate token ID {idx} for {tok!r}")
            self._vocab[tok], self._id_to_token[idx] = idx, tok

        for byte_sym in self._b2u.values():
            if byte_sym not in self._vocab:
                raise SerializationError(f"vocab missing base byte symbol {byte_sym!r}")

        self._merge_ranks: Dict[Tuple[str, str], int] = {}
        for rank, pair in enumerate(merges):
            if not isinstance(pair, tuple) or len(pair) != 2 or not all(isinstance(p, str) and p for p in pair):
                raise SerializationError(f"Invalid BPE merge pair at rank {rank}: {pair!r}")
            joined = pair[0] + pair[1]
            if pair[0] not in self._vocab or pair[1] not in self._vocab or joined not in self._vocab:
                raise SerializationError(f"Merge tokens {pair!r} -> {joined!r} missing from vocab")
            self._merge_ranks.setdefault(pair, rank)

        self._special_tokens: Dict[str, int] = {}
        self._id_to_special: Dict[int, str] = {}
        for stok, sidx in (special_tokens or {}).items():
            if not isinstance(stok, str) or not stok or isinstance(sidx, bool) or not isinstance(sidx, int):
                raise SerializationError(f"Invalid special token {stok!r} -> {sidx!r}")
            existing = self._id_to_token.get(sidx)
            if (existing is not None and existing != stok) or sidx in self._id_to_special:
                raise SerializationError(f"Conflicting special token ID {sidx}")
            self._special_tokens[stok], self._id_to_special[sidx] = sidx, stok

        ordered = sorted(self._special_tokens.keys(), key=len, reverse=True)
        self._special_re = re.compile("(" + "|".join(re.escape(s) for s in ordered) + ")") if ordered else None
        self._pretokenize = bool(pretokenize)
        self._bpe_cache: Dict[str, Tuple[int, ...]] = {}

    @classmethod
    def from_merges(
        cls, merges: Sequence[Tuple[str, str]], *,
        special_tokens: Optional[Mapping[str, int]] = None, pretokenize: bool = True,
    ) -> ByteLevelBPE:
        """Build a `ByteLevelBPE` starting from the 256 byte vocabulary plus `merges`."""
        vocab = build_base_byte_vocab()
        for left, right in merges:
            vocab.setdefault(left + right, len(vocab))
        return cls(vocab, merges, special_tokens=dict(special_tokens) if special_tokens else None, pretokenize=pretokenize)

    @classmethod
    def train_from_texts(
        cls, texts: Iterable[str], vocab_size: int = 384, *,
        special_tokens: Optional[Sequence[str]] = None, pretokenize: bool = True,
    ) -> ByteLevelBPE:
        """Deterministically train a reference byte-level BPE from sample strings."""
        num_special = len(special_tokens) if special_tokens else 0
        if vocab_size < 256 + num_special:
            raise SerializationError("vocab_size must accommodate 256 byte tokens + special_tokens")
        b2u = bytes_to_unicode()
        word_counts: Dict[Tuple[str, ...], int] = {}
        for text in texts:
            for piece in (_PRETOKENIZE_RE.findall(text) if pretokenize else ([text] if text else [])):
                syms = tuple(b2u[b] for b in piece.encode("utf-8"))
                if syms:
                    word_counts[syms] = word_counts.get(syms, 0) + 1

        vocab, merges = build_base_byte_vocab(), []
        for _ in range(vocab_size - 256 - num_special):
            pair_freq: Dict[Tuple[str, str], int] = {}
            for word, count in word_counts.items():
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    pair_freq[pair] = pair_freq.get(pair, 0) + count
            if not pair_freq:
                break
            best_pair = max(pair_freq.items(), key=lambda kv: (kv[1], kv[0]))[0]
            merged_tok = best_pair[0] + best_pair[1]
            vocab.setdefault(merged_tok, len(vocab))
            merges.append(best_pair)
            updated: Dict[Tuple[str, ...], int] = {}
            for word, count in word_counts.items():
                new_w, i = [], 0
                while i < len(word):
                    if i + 1 < len(word) and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                        new_w.append(merged_tok)
                        i += 2
                    else:
                        new_w.append(word[i])
                        i += 1
                tup = tuple(new_w)
                updated[tup] = updated.get(tup, 0) + count
            word_counts = updated

        spec_map = {s: len(vocab) + i for i, s in enumerate(special_tokens)} if special_tokens else None
        return cls(vocab, merges, special_tokens=spec_map, pretokenize=pretokenize)

    @property
    def vocab_size(self) -> int:
        return len(set(self._id_to_token.keys()) | set(self._id_to_special.keys()))

    def _merge_piece(self, mapped_piece: str) -> Tuple[int, ...]:
        if (cached := self._bpe_cache.get(mapped_piece)) is not None:
            return cached
        symbols = list(mapped_piece)
        while len(symbols) >= 2:
            best_rank: Optional[int] = None
            best_pair: Optional[Tuple[str, str]] = None
            for i in range(len(symbols) - 1):
                cand = (symbols[i], symbols[i + 1])
                rank = self._merge_ranks.get(cand)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_pair = rank, cand
            if best_pair is None:
                break
            merged_sym, next_syms, idx, n = best_pair[0] + best_pair[1], [], 0, len(symbols)
            while idx < n:
                if idx + 1 < n and symbols[idx] == best_pair[0] and symbols[idx + 1] == best_pair[1]:
                    next_syms.append(merged_sym)
                    idx += 2
                else:
                    next_syms.append(symbols[idx])
                    idx += 1
            symbols = next_syms
        ids = tuple(self._vocab[s] for s in symbols)
        if len(self._bpe_cache) < 16384:
            self._bpe_cache[mapped_piece] = ids
        return ids

    def _encode_plain_span(self, text: str) -> List[int]:
        out: List[int] = []
        for piece in (_PRETOKENIZE_RE.findall(text) if self._pretokenize else ([text] if text else [])):
            out.extend(self._merge_piece("".join(self._b2u[b] for b in piece.encode("utf-8"))))
        return out

    def encode(self, text: str, *, allow_special: bool = True) -> List[int]:
        """Encode `text` into deterministic BPE token IDs."""
        if not isinstance(text, str):
            raise TypeError(f"Expected str for encode(), got {type(text).__name__}")
        try:
            text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise SerializationError("Input text contains invalid UTF-8 surrogates") from exc
        if not text:
            return []
        if not allow_special or self._special_re is None:
            return self._encode_plain_span(text)
        token_ids: List[int] = []
        for seg in self._special_re.split(text):
            if seg:
                spec_id = self._special_tokens.get(seg)
                if spec_id is not None:
                    token_ids.append(spec_id)
                else:
                    token_ids.extend(self._encode_plain_span(seg))
        return token_ids

    def decode(self, token_ids: Sequence[int], *, errors: str = "strict") -> str:
        """Decode BPE token IDs back into a UTF-8 Python string."""
        chunks, byte_buf = [], bytearray()
        for tid in token_ids:
            if isinstance(tid, bool) or not isinstance(tid, int):
                raise SerializationError(f"Token ID must be int, got {tid!r}")
            if tid in self._id_to_special:
                if byte_buf:
                    chunks.append(byte_buf.decode("utf-8", errors=errors))
                    byte_buf.clear()
                chunks.append(self._id_to_special[tid])
                continue
            tok = self._id_to_token.get(tid)
            if tok is None:
                raise SerializationError(f"Unknown token ID: {tid}")
            for ch in tok:
                if (bval := self._u2b.get(ch)) is None:
                    raise SerializationError(f"Non-byte BPE symbol in vocab: {ch!r}")
                byte_buf.append(bval)
        if byte_buf:
            chunks.append(byte_buf.decode("utf-8", errors=errors))
        return "".join(chunks)

    def count_tokens(self, text: str, *, allow_special: bool = True) -> int:
        return len(self.encode(text, allow_special=allow_special))


def _normalize_json_value(value: Any, *, path: str, in_schema_required: bool = False) -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ToolSchemaError(f"Non-finite float at {path}: {value!r}")
        return 0.0 if value == 0.0 else value
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ToolSchemaError(f"Invalid UTF-8 string at {path}") from exc
        return value
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for k in sorted(value.keys(), key=str):
            if not isinstance(k, str):
                raise ToolSchemaError(f"Non-string key {k!r} at {path}")
            is_req = k == "required" and isinstance(value[k], Sequence) and not isinstance(value[k], str)
            out[k] = _normalize_json_value(value[k], path=f"{path}.{k}", in_schema_required=is_req)
        return out
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, memoryview)):
        items = [_normalize_json_value(v, path=f"{path}[{i}]") for i, v in enumerate(value)]
        if in_schema_required:
            if not all(isinstance(x, str) for x in items):
                raise ToolSchemaError(f"JSON Schema 'required' entries must be strings at {path}")
            return sorted(set(items))
        return items
    raise ToolSchemaError(f"Unsupported JSON value of type {type(value).__name__} at {path}")


@dataclasses.dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments_json: str

    @classmethod
    def create(cls, call_id: str, name: str, arguments: Union[str, Mapping[str, Any]]) -> ToolCall:
        if not isinstance(call_id, str) or not call_id.strip():
            raise ConversationValidationError("ToolCall.call_id must be a non-empty string")
        if not isinstance(name, str) or not _IDENTIFIER_RE.match(name):
            raise ToolSchemaError(f"Invalid tool name in ToolCall: {name!r}")
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ToolSchemaError(f"ToolCall {call_id!r} arguments invalid JSON: {exc}") from exc
        elif isinstance(arguments, Mapping):
            parsed = arguments
        else:
            raise ToolSchemaError("ToolCall arguments must be a JSON object or JSON string")
        if not isinstance(parsed, Mapping):
            raise ToolSchemaError("ToolCall arguments must decode to a JSON object")
        clean = _normalize_json_value(parsed, path=f"$.tool_call({call_id})")
        return cls(call_id.strip(), name, json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))


@dataclasses.dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: Dict[str, Any]


@dataclasses.dataclass(frozen=True)
class Message:
    role: str
    content: str = ""
    tool_calls: Tuple[ToolCall, ...] = ()
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> Message:
        if not isinstance(raw, Mapping):
            raise ConversationValidationError("Message entry must be a mapping")
        role = raw.get("role")
        if not isinstance(role, str) or role not in VALID_ROLES:
            raise ConversationValidationError(f"Invalid message role: {role!r}")
        raw_content = "" if raw.get("content") is None else raw.get("content", "")
        if not isinstance(raw_content, str):
            raise ConversationValidationError(f"Message content must be str, got {type(raw_content).__name__}")
        try:
            raw_content.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ConversationValidationError("Message content has invalid UTF-8") from exc
        raw_calls = raw.get("tool_calls") or ()
        if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, (str, bytes)):
            raise ConversationValidationError("tool_calls must be a sequence")
        calls: List[ToolCall] = []
        for item in raw_calls:
            if isinstance(item, ToolCall):
                calls.append(ToolCall.create(item.call_id, item.name, item.arguments_json))
            elif isinstance(item, Mapping):
                src = item["function"] if isinstance(item.get("function"), Mapping) else item
                calls.append(ToolCall.create(item.get("call_id", item.get("id", "")), src.get("name", ""), src.get("arguments", {})))
            else:
                raise ConversationValidationError(f"Invalid tool_call entry: {item!r}")
        tc_id = raw.get("tool_call_id")
        if tc_id is not None and (not isinstance(tc_id, str) or not tc_id.strip()):
            raise ConversationValidationError("tool_call_id must be a non-empty string")
        msg_name = raw.get("name")
        if msg_name is not None and (not isinstance(msg_name, str) or not msg_name.strip()):
            raise ConversationValidationError("Optional message name must be a non-empty string")
        return cls(role, raw_content, tuple(calls), tc_id.strip() if isinstance(tc_id, str) else None, msg_name.strip() if isinstance(msg_name, str) else None)


def canonicalize_tool_schema(tool: Union[ToolDefinition, Mapping[str, Any]]) -> Dict[str, Any]:
    """Validate and canonicalize a single tool definition into a sorted dict."""
    if isinstance(tool, ToolDefinition):
        name, description, parameters = tool.name, tool.description, tool.parameters
    elif isinstance(tool, Mapping):
        src = tool["function"] if isinstance(tool.get("function"), Mapping) else tool
        name, description = src.get("name"), src.get("description", "")
        parameters = src.get("parameters", {"type": "object", "properties": {}})
    else:
        raise ToolSchemaError(f"Invalid tool definition type: {type(tool).__name__}")
    if not isinstance(name, str) or not _IDENTIFIER_RE.match(name):
        raise ToolSchemaError(f"Invalid tool name: {name!r}")
    if not isinstance(description, str) or not isinstance(parameters, Mapping):
        raise ToolSchemaError(f"Tool {name!r} requires string description and mapping parameters")
    norm_params = _normalize_json_value(parameters, path=f"$.tools.{name}.parameters")
    if norm_params.get("type", "object") != "object":
        raise ToolSchemaError(f"Tool {name!r} root schema type must be 'object'")
    return {"description": description.strip(), "name": name, "parameters": norm_params}


def canonicalize_tools(tools: Optional[Sequence[Union[ToolDefinition, Mapping[str, Any]]]]) -> Tuple[str, str, Set[str]]:
    """Canonicalize and sort tool definitions, returning `(json_str, sha256, tool_names)`."""
    if not tools:
        return "[]", hashlib.sha256(b"[]").hexdigest(), set()
    seen_names: Set[str] = set()
    canonical_list: List[Dict[str, Any]] = []
    for item in tools:
        canon = canonicalize_tool_schema(item)
        if canon["name"] in seen_names:
            raise ToolSchemaError(f"Duplicate tool definition name: {canon['name']!r}")
        seen_names.add(canon["name"])
        canonical_list.append(canon)
    canonical_list.sort(key=lambda entry: entry["name"])
    serialized = json.dumps(canonical_list, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return serialized, hashlib.sha256(serialized.encode("utf-8")).hexdigest(), seen_names


def normalize_and_validate_conversation(
    messages: Sequence[Union[Message, Mapping[str, Any]]], *, defined_tool_names: Optional[Set[str]] = None,
) -> Tuple[Message, ...]:
    """Validate role transitions and 1-to-1 `tool_call_id` resolution."""
    if not isinstance(messages, Sequence) or not messages:
        raise ConversationValidationError("Conversation must be a non-empty sequence of messages")
    parsed = [m if isinstance(m, Message) else Message.from_mapping(m) for m in messages]
    canonical_msgs: List[Message] = []
    seen_call_ids: Set[str] = set()
    idx, n = 0, len(parsed)

    while idx < n and parsed[idx].role == "system":
        if parsed[idx].tool_calls or parsed[idx].tool_call_id is not None:
            raise ConversationValidationError("System message cannot carry tool_calls or tool_call_id")
        canonical_msgs.append(parsed[idx])
        idx += 1
    if idx >= n or parsed[idx].role != "user":
        raise ConversationValidationError("Dialogue must begin with a 'user' message after system preamble")

    expected_next: Set[str] = {"user"}
    while idx < n:
        msg = parsed[idx]
        if msg.role == "system" or msg.role not in expected_next:
            raise ConversationValidationError(f"Invalid role transition at index {idx}: got {msg.role!r}")
        if msg.role == "user":
            if msg.tool_calls or msg.tool_call_id is not None:
                raise ConversationValidationError(f"User message at {idx} cannot carry tool fields")
            canonical_msgs.append(msg)
            expected_next = {"assistant"}
            idx += 1
            continue
        if msg.role == "assistant":
            if msg.tool_call_id is not None:
                raise ConversationValidationError(f"Assistant message at {idx} cannot carry tool_call_id")
            if not msg.tool_calls:
                canonical_msgs.append(msg)
                expected_next = {"user"}
                idx += 1
                continue
            rebuilt_calls = tuple(ToolCall.create(tc.call_id, tc.name, tc.arguments_json) for tc in msg.tool_calls)
            expected_ids: List[str] = []
            for tc in rebuilt_calls:
                if tc.call_id in seen_call_ids:
                    raise ConversationValidationError(f"Duplicate tool_call_id {tc.call_id!r} at {idx}")
                if defined_tool_names is not None and tc.name not in defined_tool_names:
                    raise ConversationValidationError(f"Assistant called undefined tool {tc.name!r}")
                seen_call_ids.add(tc.call_id)
                expected_ids.append(tc.call_id)
            canonical_msgs.append(Message("assistant", msg.content, rebuilt_calls, None, msg.name))
            idx += 1
            results_by_id: Dict[str, Message] = {}
            expected_set = set(expected_ids)
            while idx < n and parsed[idx].role == "tool":
                tmsg = parsed[idx]
                if tmsg.tool_calls or tmsg.tool_call_id is None:
                    raise ConversationValidationError(f"Malformed tool result message at index {idx}")
                if tmsg.tool_call_id not in expected_set or tmsg.tool_call_id in results_by_id:
                    raise ConversationValidationError(f"Orphan or duplicate tool_call_id {tmsg.tool_call_id!r}")
                results_by_id[tmsg.tool_call_id] = tmsg
                idx += 1
            if missing := [cid for cid in expected_ids if cid not in results_by_id]:
                raise ConversationValidationError(f"Unresolved tool_call_id(s): {missing!r}")
            canonical_msgs.extend(results_by_id[cid] for cid in expected_ids)
            expected_next = {"assistant"}
            continue
        raise ConversationValidationError(f"Orphan 'tool' message at index {idx}")

    return tuple(canonical_msgs)


@dataclasses.dataclass(frozen=True)
class FitResult:
    messages: Tuple[Message, ...]
    serialized_prompt: str
    token_ids: Tuple[int, ...]
    token_count: int
    dropped_turn_groups: int
    dropped_messages: int
    tools_sha256: str
    prompt_sha256: str


class PromptSerializer:
    """Canonical multi-turn serializer and atomic turn-group context fitter."""

    def __init__(self, tokenizer: Optional[ByteLevelBPE] = None) -> None:
        spec_ids = {tok: 256 + idx for idx, tok in enumerate(_SPECIAL_MARKERS)}
        self._tokenizer = tokenizer or ByteLevelBPE(build_base_byte_vocab(), (), special_tokens=spec_ids)

    @property
    def tokenizer(self) -> ByteLevelBPE:
        return self._tokenizer

    @staticmethod
    def _format_message(msg: Message) -> str:
        hdr = [f"role={msg.role}"]
        if msg.name is not None:
            hdr.append(f"name={json.dumps(msg.name, ensure_ascii=False)}")
        if msg.tool_call_id is not None:
            hdr.append(f"tool_call_id={json.dumps(msg.tool_call_id, ensure_ascii=False)}")
        lines = [f"<|msg|>{' '.join(hdr)}", msg.content]
        if msg.tool_calls:
            calls_payload = json.dumps(
                [{"arguments": json.loads(tc.arguments_json), "call_id": tc.call_id, "name": tc.name} for tc in msg.tool_calls],
                sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
            )
            lines.append(f"<|tool_calls|>{calls_payload}<|/tool_calls|>")
        lines.append("<|/msg|>")
        return "\n".join(lines)

    def _render_prompt(self, msgs: Sequence[Message], tools_json: str, has_tools: bool) -> str:
        blocks = [f"<|tools|>\n{tools_json}\n<|/tools|>"] if has_tools else []
        blocks.extend(self._format_message(m) for m in msgs)
        return "\n".join(blocks) + "\n"

    def serialize(
        self, messages: Sequence[Union[Message, Mapping[str, Any]]], *,
        tools: Optional[Sequence[Union[ToolDefinition, Mapping[str, Any]]]] = None,
    ) -> Tuple[str, str, str, Tuple[Message, ...]]:
        """Validate and render canonical prompt text, returning `(text, tools_sha, prompt_sha, msgs)`."""
        tools_json, tools_sha, defined_names = canonicalize_tools(tools)
        validated = normalize_and_validate_conversation(messages, defined_tool_names=defined_names if tools is not None else None)
        rendered = self._render_prompt(validated, tools_json, bool(tools))
        return rendered, tools_sha, hashlib.sha256(rendered.encode("utf-8")).hexdigest(), validated

    @staticmethod
    def _segment_dialogue(
        validated: Tuple[Message, ...],
    ) -> Tuple[List[Message], List[List[Message]], Message, List[List[Message]], Optional[Message]]:
        system_msgs: List[Message] = []
        idx, n = 0, len(validated)
        while idx < n and validated[idx].role == "system":
            system_msgs.append(validated[idx])
            idx += 1
        cycles: List[List[Message]] = []
        cur: List[Message] = []
        while idx < n:
            if validated[idx].role == "user":
                if cur:
                    cycles.append(cur)
                cur = [validated[idx]]
            else:
                cur.append(validated[idx])
            idx += 1
        if cur:
            cycles.append(cur)

        latest = cycles[-1]
        tail_steps, tail_final, j = [], None, 1
        while j < len(latest):
            if latest[j].role == "assistant" and latest[j].tool_calls:
                span = 1 + len(latest[j].tool_calls)
                tail_steps.append(list(latest[j : j + span]))
                j += span
            else:
                tail_final = latest[j]
                j += 1
        return system_msgs, cycles[:-1], latest[0], tail_steps, tail_final

    def fit_to_budget(
        self, messages: Sequence[Union[Message, Mapping[str, Any]]], *, max_tokens: int,
        policy: str = "sliding_window",
        tools: Optional[Sequence[Union[ToolDefinition, Mapping[str, Any]]]] = None,
    ) -> FitResult:
        """Fit conversation within `max_tokens` while preserving atomic assistant+tool pairs."""
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise SerializationError("max_tokens must be a positive integer")
        if policy not in TRUNCATION_POLICIES:
            raise SerializationError(f"Unknown truncation policy: {policy!r}")

        tools_json, tools_sha, defined_names = canonicalize_tools(tools)
        enforce_names = defined_names if tools is not None else None
        validated = normalize_and_validate_conversation(messages, defined_tool_names=enforce_names)
        sys_msgs, hist_cycles, latest_user, tail_steps, tail_final = self._segment_dialogue(validated)
        kept_hist, kept_steps, kept_final, dropped_groups = list(hist_cycles), list(tail_steps), tail_final, 0

        while True:
            assembled = [*sys_msgs, *(m for g in kept_hist for m in g), latest_user, *(m for s in kept_steps for m in s)]
            if kept_final is not None:
                assembled.append(kept_final)
            rendered = self._render_prompt(assembled, tools_json, bool(tools))
            tids = self._tokenizer.encode(rendered, allow_special=True)
            if len(tids) <= max_tokens:
                final_msgs = normalize_and_validate_conversation(assembled, defined_tool_names=enforce_names)
                return FitResult(
                    final_msgs, rendered, tuple(tids), len(tids), dropped_groups,
                    len(validated) - len(final_msgs), tools_sha,
                    hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
                )
            if kept_hist:
                kept_hist.pop(0 if policy == "sliding_window" else (len(kept_hist) // 2 if len(kept_hist) >= 2 else 0))
            elif kept_steps:
                kept_steps.pop(0 if policy == "sliding_window" else (len(kept_steps) // 2 if len(kept_steps) >= 2 else 0))
            elif kept_final is not None:
                kept_final = None
            else:
                raise BudgetExceededError(f"Pinned context + latest user turn requires {len(tids)} tokens > {max_tokens}")
            dropped_groups += 1
