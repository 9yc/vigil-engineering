"""Unit tests for runtime_support.prompt_serializer."""

from __future__ import annotations

import unittest

from runtime_support.prompt_serializer import (
    BudgetExceededError,
    ByteLevelBPE,
    ConversationValidationError,
    Message,
    PromptSerializer,
    SerializationError,
    ToolCall,
    ToolDefinition,
    ToolSchemaError,
    bytes_to_unicode,
    canonicalize_tools,
    normalize_and_validate_conversation,
)


class PromptSerializerTest(unittest.TestCase):
    """Tests for byte-level BPE, schema canonicalization, validation, and truncation."""

    def test_bpe_rank_priority_merging_and_utf8_multibyte_roundtrip(self) -> None:
        b2u = bytes_to_unicode()
        a, b, c = b2u[ord("a")], b2u[ord("b")], b2u[ord("c")]

        # Rank 0 merges (a, b) -> "ab"; Rank 1 merges (b, c) -> "bc"; Rank 2 merges ("ab", c) -> "abc".
        bpe_ab = ByteLevelBPE.from_merges([(a, b), (b, c), (a + b, c)], pretokenize=False)
        ids_abc = bpe_ab.encode("abc")
        self.assertEqual(len(ids_abc), 1)
        self.assertEqual(bpe_ab.decode(ids_abc), "abc")

        # Flip merge priority so (b, c) has rank 0 -> ("a", "bc") = 2 tokens.
        bpe_bc = ByteLevelBPE.from_merges([(b, c), (a, b), (a + b, c)], pretokenize=False)
        ids_bc = bpe_bc.encode("abc")
        self.assertEqual(len(ids_bc), 2)
        self.assertEqual(bpe_bc.decode(ids_bc), "abc")

        corpus = [
            "deterministic systems engineering serializer utility",
            "latency throughput quantiles verification",
            "UTF-8 多字节 テスト 🧪 🚀 ∑∏∫ café naïve",
        ]
        trained = ByteLevelBPE.train_from_texts(
            corpus, vocab_size=340, special_tokens=["<|tools|>", "<|/tools|>", "<|msg|>", "<|/msg|>"]
        )
        samples = [
            "",
            "hello world! 12345",
            "多字节 UTF-8 roundtrip 🚀🧪 café ∑x_i",
            "<|msg|>role=user\nПривет мир 🌍<|/msg|>",
            "\n\t  leading and trailing whitespace  \r\n",
        ]
        for text in samples:
            with self.subTest(text=text):
                encoded = trained.encode(text, allow_special=True)
                self.assertEqual(trained.count_tokens(text, allow_special=True), len(encoded))
                self.assertEqual(trained.decode(encoded), text)

        phrase = "deterministic systems engineering serializer"
        self.assertLess(trained.count_tokens(phrase), len(phrase.encode("utf-8")))

    def test_canonical_tool_schema_hashing_invariance_under_reordering(self) -> None:
        tools_order_1 = [
            {
                "name": "query_metrics",
                "description": "Fetch time-series samples.",
                "parameters": {
                    "type": "object",
                    "required": ["metric", "window_s", "metric"],
                    "properties": {
                        "window_s": {"type": "integer", "minimum": 1},
                        "metric": {"type": "string", "description": "Series identifier"},
                    },
                },
            },
            ToolDefinition(
                name="inspect_node",
                description="Read host status.",
                parameters={
                    "properties": {"node_id": {"type": "string"}, "verbose": {"type": "boolean"}},
                    "required": ["node_id"],
                    "type": "object",
                },
            ),
        ]
        tools_order_2 = [
            {
                "function": {
                    "description": "Read host status.",
                    "parameters": {
                        "type": "object",
                        "required": ["node_id"],
                        "properties": {
                            "verbose": {"type": "boolean"},
                            "node_id": {"type": "string"},
                        },
                    },
                    "name": "inspect_node",
                }
            },
            {
                "parameters": {
                    "properties": {
                        "metric": {"description": "Series identifier", "type": "string"},
                        "window_s": {"minimum": 1, "type": "integer"},
                    },
                    "type": "object",
                    "required": ["window_s", "metric"],
                },
                "description": "Fetch time-series samples.",
                "name": "query_metrics",
            },
        ]

        json_1, sha_1, names_1 = canonicalize_tools(tools_order_1)
        json_2, sha_2, _ = canonicalize_tools(tools_order_2)
        self.assertEqual(json_1, json_2)
        self.assertEqual(sha_1, sha_2)
        self.assertEqual(names_1, {"query_metrics", "inspect_node"})

        with self.assertRaises(ToolSchemaError):
            canonicalize_tools([tools_order_1[0], tools_order_1[0]])
        with self.assertRaises(ToolSchemaError):
            canonicalize_tools([{
                "name": "bad_float", "description": "invalid",
                "parameters": {"type": "object", "minimum": float("nan")},
            }])

    def test_invalid_role_transitions_and_orphan_tool_call_id_rejection(self) -> None:
        invalid_cases = [
            [{"role": "assistant", "content": "hi"}],
            [{"role": "user", "content": "first"}, {"role": "user", "content": "second"}],
            [{"role": "user", "content": "hello"}, {"role": "system", "content": "late"}],
            [
                {"role": "user", "content": "check status"},
                {
                    "role": "assistant", "content": "",
                    "tool_calls": [
                        {"call_id": "call-1", "name": "ping", "arguments": {"host": "a"}},
                        {"call_id": "call-2", "name": "ping", "arguments": {"host": "b"}},
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": '{"ok":true}'},
            ],
            [
                {"role": "user", "content": "check status"},
                {"role": "tool", "tool_call_id": "orphan-99", "content": "{}"},
            ],
            [
                {"role": "user", "content": "check"},
                {"role": "assistant", "content": "", "tool_calls": [{"call_id": "c1", "name": "ping", "arguments": {}}]},
                {"role": "tool", "tool_call_id": "wrong-id", "content": "{}"},
            ],
        ]
        for case in invalid_cases:
            with self.subTest(case=case):
                with self.assertRaises(ConversationValidationError):
                    normalize_and_validate_conversation(case)

        with self.assertRaises(ConversationValidationError):
            normalize_and_validate_conversation(
                [
                    {"role": "user", "content": "check"},
                    {"role": "assistant", "content": "", "tool_calls": [{"call_id": "c1", "name": "undeclared", "arguments": {}}]},
                    {"role": "tool", "tool_call_id": "c1", "content": "{}"},
                ],
                defined_tool_names={"allowed_tool"},
            )

        normalized = normalize_and_validate_conversation(
            [
                {"role": "system", "content": "Pinned instructions."},
                {"role": "user", "content": "Probe both endpoints."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        ToolCall.create("c-alpha", "probe", {"target": "a", "timeout": 2}),
                        ToolCall.create("c-beta", "probe", {"timeout": 2, "target": "b"}),
                    ],
                },
                {"role": "tool", "tool_call_id": "c-beta", "content": "beta_ok"},
                {"role": "tool", "tool_call_id": "c-alpha", "content": "alpha_ok"},
                {"role": "assistant", "content": "Both endpoints responded."},
            ],
            defined_tool_names={"probe"},
        )
        self.assertEqual([m.tool_call_id for m in normalized[3:5]], ["c-alpha", "c-beta"])

    @staticmethod
    def _build_multi_turn_fixture() -> tuple[list[dict], list[dict]]:
        tools = [{
            "name": "lookup_stat", "description": "Retrieve metric counter.",
            "parameters": {"type": "object", "required": ["key"], "properties": {"key": {"type": "string"}}},
        }]
        messages: list[dict] = [
            {"role": "system", "content": "Always report exact integer units."},
            {"role": "user", "content": "Turn-1 question: inspect alpha and beta counters."},
            {
                "role": "assistant", "content": "",
                "tool_calls": [
                    {"call_id": "t1-a", "name": "lookup_stat", "arguments": {"key": "alpha"}},
                    {"call_id": "t1-b", "name": "lookup_stat", "arguments": {"key": "beta"}},
                ],
            },
            {"role": "tool", "tool_call_id": "t1-a", "content": '{"val": 101}'},
            {"role": "tool", "tool_call_id": "t1-b", "content": '{"val": 202}'},
            {"role": "assistant", "content": "Turn-1 answer: alpha=101, beta=202."},
            {"role": "user", "content": "Turn-2 question: inspect gamma counter with extra padding."},
            {"role": "assistant", "content": "", "tool_calls": [{"call_id": "t2-g", "name": "lookup_stat", "arguments": {"key": "gamma"}}]},
            {"role": "tool", "tool_call_id": "t2-g", "content": '{"val": 303, "note": "steady"}'},
            {"role": "assistant", "content": "Turn-2 answer: gamma=303."},
            {"role": "user", "content": "Turn-3 question: inspect delta counter."},
            {"role": "assistant", "content": "", "tool_calls": [{"call_id": "t3-d", "name": "lookup_stat", "arguments": {"key": "delta"}}]},
            {"role": "tool", "tool_call_id": "t3-d", "content": '{"val": 404}'},
            {"role": "assistant", "content": "Turn-3 answer: delta=404."},
            {"role": "user", "content": "Turn-4 latest question: compute delta minus alpha."},
        ]
        return messages, tools

    def _assert_no_stranded_tool_pairs(self, msgs: tuple[Message, ...]) -> None:
        revalidated = normalize_and_validate_conversation(msgs, defined_tool_names={"lookup_stat"})
        self.assertEqual(revalidated, msgs)
        declared_ids = {tc.call_id for m in msgs if m.role == "assistant" for tc in m.tool_calls}
        result_ids = {m.tool_call_id for m in msgs if m.role == "tool"}
        self.assertEqual(declared_ids, result_ids)

    def test_sliding_window_and_middle_out_preserve_atomic_tool_pairs(self) -> None:
        messages, tools = self._build_multi_turn_fixture()
        bpe = ByteLevelBPE.train_from_texts(
            [m.get("content", "") for m in messages],
            vocab_size=320,
            special_tokens=["<|tools|>", "<|/tools|>", "<|msg|>", "<|/msg|>", "<|tool_calls|>", "<|/tool_calls|>"],
        )
        serializer = PromptSerializer(bpe)
        full_fit = serializer.fit_to_budget(messages, max_tokens=10_000, tools=tools)
        self.assertEqual(full_fit.dropped_turn_groups, 0)
        # Re-serializing identical messages produces identical SHA-256 prompt digest.
        text_a, tsha_a, psha_a, _ = serializer.serialize(messages, tools=tools)
        text_b, tsha_b, psha_b, _ = serializer.serialize(messages, tools=tools)
        self.assertEqual((text_a, tsha_a, psha_a), (text_b, tsha_b, psha_b))
        self.assertEqual(psha_a, full_fit.prompt_sha256)

        budget_drop_one = int(full_fit.token_count * 0.82)

        sw_one = serializer.fit_to_budget(
            messages, max_tokens=budget_drop_one, policy="sliding_window", tools=tools
        )
        self.assertLessEqual(sw_one.token_count, budget_drop_one)
        self.assertGreaterEqual(sw_one.dropped_turn_groups, 1)
        self._assert_no_stranded_tool_pairs(sw_one.messages)
        self.assertEqual(sw_one.messages[0].role, "system")
        self.assertIn("Turn-4 latest", sw_one.messages[-1].content)
        sw_contents = "\n".join(m.content for m in sw_one.messages)
        self.assertNotIn("Turn-1 question", sw_contents)
        self.assertIn("Turn-3 question", sw_contents)

        mo_one = serializer.fit_to_budget(
            messages, max_tokens=budget_drop_one, policy="middle_out", tools=tools
        )
        self.assertLessEqual(mo_one.token_count, budget_drop_one)
        self.assertEqual(mo_one.dropped_turn_groups, 1)
        self._assert_no_stranded_tool_pairs(mo_one.messages)
        self.assertEqual(mo_one.messages[0].role, "system")
        self.assertIn("Turn-4 latest", mo_one.messages[-1].content)
        mo_contents = "\n".join(m.content for m in mo_one.messages)
        self.assertIn("Turn-1 question", mo_contents)
        self.assertNotIn("Turn-2 question", mo_contents)
        self.assertIn("Turn-3 question", mo_contents)

        active_loop = [
            {"role": "system", "content": "Pinned system."},
            {"role": "user", "content": "Active multi-step request."},
            {"role": "assistant", "content": "", "tool_calls": [{"call_id": "s1", "name": "lookup_stat", "arguments": {"key": "k1"}}]},
            {"role": "tool", "tool_call_id": "s1", "content": '{"v": 1}'},
            {"role": "assistant", "content": "", "tool_calls": [{"call_id": "s2", "name": "lookup_stat", "arguments": {"key": "k2"}}]},
            {"role": "tool", "tool_call_id": "s2", "content": '{"v": 2}'},
        ]
        full_loop = serializer.fit_to_budget(active_loop, max_tokens=5000, tools=tools)
        trimmed_loop = serializer.fit_to_budget(
            active_loop, max_tokens=full_loop.token_count - 15, policy="sliding_window", tools=tools
        )
        self._assert_no_stranded_tool_pairs(trimmed_loop.messages)
        self.assertEqual([tc.call_id for m in trimmed_loop.messages for tc in m.tool_calls], ["s2"])

        with self.assertRaises(BudgetExceededError):
            serializer.fit_to_budget(messages, max_tokens=15, tools=tools)
        with self.assertRaises(SerializationError):
            serializer.fit_to_budget(messages, max_tokens=500, policy="unknown_policy")


if __name__ == "__main__":
    unittest.main()
