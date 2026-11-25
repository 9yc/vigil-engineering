#include "slab_arena.hpp"

#include <cctype>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

struct JsonObj {
  std::map<std::string, std::string> str_fields;
  std::map<std::string, uint64_t> uint_fields;
};

class LineParser {
 public:
  explicit LineParser(const std::string& src) : src_(src) {}

  JsonObj ParseObject() {
    SkipWs();
    Expect('{');
    JsonObj obj;
    SkipWs();
    if (TryConsume('}')) return obj;
    while (true) {
      SkipWs();
      std::string key = ParseString();
      SkipWs();
      Expect(':');
      SkipWs();
      if (pos_ >= src_.size()) throw std::runtime_error("Unexpected end of JSON line");
      if (src_[pos_] == '"') {
        obj.str_fields[key] = ParseString();
      } else if (std::isdigit(static_cast<unsigned char>(src_[pos_]))) {
        obj.uint_fields[key] = ParseUint64();
      } else {
        throw std::runtime_error("Unsupported or negative token for field: " + key);
      }
      SkipWs();
      if (TryConsume('}')) break;
      Expect(',');
    }
    SkipWs();
    if (pos_ != src_.size()) throw std::runtime_error("Trailing characters in JSON line");
    return obj;
  }

 private:
  void SkipWs() {
    while (pos_ < src_.size() && std::isspace(static_cast<unsigned char>(src_[pos_]))) ++pos_;
  }
  void Expect(char ch) {
    if (pos_ >= src_.size() || src_[pos_] != ch) {
      throw std::runtime_error(std::string("Expected '") + ch + "'");
    }
    ++pos_;
  }
  bool TryConsume(char ch) {
    if (pos_ < src_.size() && src_[pos_] == ch) {
      ++pos_;
      return true;
    }
    return false;
  }
  std::string ParseString() {
    Expect('"');
    std::string out;
    while (pos_ < src_.size()) {
      char c = src_[pos_++];
      if (c == '"') return out;
      if (c == '\\') {
        if (pos_ >= src_.size()) throw std::runtime_error("Invalid string escape");
        char esc = src_[pos_++];
        if (esc == '"' || esc == '\\' || esc == '/') out.push_back(esc);
        else if (esc == 'n') out.push_back('\n');
        else if (esc == 'r') out.push_back('\r');
        else if (esc == 't') out.push_back('\t');
        else throw std::runtime_error("Unsupported escape sequence");
      } else {
        out.push_back(c);
      }
    }
    throw std::runtime_error("Unterminated string literal");
  }
  uint64_t ParseUint64() {
    uint64_t val = 0;
    while (pos_ < src_.size() && std::isdigit(static_cast<unsigned char>(src_[pos_]))) {
      const uint64_t d = static_cast<uint64_t>(src_[pos_++] - '0');
      if (val > (std::numeric_limits<uint64_t>::max() - d) / 10) {
        throw std::runtime_error("Integer overflow in JSON line");
      }
      val = val * 10 + d;
    }
    return val;
  }

  const std::string& src_;
  size_t pos_ = 0;
};

struct TraceEvent {
  size_t line_no = 0;
  std::string op;
  std::string id;
  bool has_raw_handle = false;
  uint64_t raw_handle = 0;
  uint64_t size = 0;
  uint64_t align = 1;
};

std::string EscapeJson(const std::string& s) {
  std::ostringstream oss;
  for (char c : s) {
    if (c == '"') oss << "\\\"";
    else if (c == '\\') oss << "\\\\";
    else if (c == '\n') oss << "\\n";
    else oss << c;
  }
  return oss.str();
}

slab_arena::Strategy ParseStrategy(const std::string& name) {
  if (name == "freelist_first_fit") return slab_arena::Strategy::kFreeListFirstFit;
  if (name == "freelist_best_fit") return slab_arena::Strategy::kFreeListBestFit;
  if (name == "buddy") return slab_arena::Strategy::kBuddy;
  throw std::runtime_error("Unknown strategy: " + name);
}

std::vector<TraceEvent> LoadTrace(const std::string& path) {
  std::ifstream in(path);
  if (!in.is_open()) throw std::runtime_error("Cannot open trace file: " + path);
  std::vector<TraceEvent> events;
  std::string line;
  size_t line_no = 0;
  while (std::getline(in, line)) {
    ++line_no;
    size_t first = line.find_first_not_of(" \t\r\n");
    if (first == std::string::npos || line[first] == '#') continue;
    JsonObj obj = LineParser(line).ParseObject();
    if (!obj.str_fields.count("op")) {
      throw std::runtime_error("Missing 'op' on line " + std::to_string(line_no));
    }
    TraceEvent ev;
    ev.line_no = line_no;
    ev.op = obj.str_fields["op"];
    if (obj.str_fields.count("id")) ev.id = obj.str_fields["id"];
    else if (obj.uint_fields.count("id")) ev.id = std::to_string(obj.uint_fields["id"]);
    if (obj.uint_fields.count("handle")) {
      ev.has_raw_handle = true;
      ev.raw_handle = obj.uint_fields["handle"];
    }
    if (ev.op == "alloc" || ev.op == "aligned_alloc") {
      if (!obj.uint_fields.count("size")) {
        throw std::runtime_error("Missing 'size' on line " + std::to_string(line_no));
      }
      ev.size = obj.uint_fields["size"];
      if (ev.op == "aligned_alloc" && !obj.uint_fields.count("align")) {
        throw std::runtime_error("Missing 'align' on line " + std::to_string(line_no));
      }
      ev.align = obj.uint_fields.count("align") ? obj.uint_fields["align"] : 1;
    } else if (ev.op == "free") {
      if (ev.id.empty() && !ev.has_raw_handle) {
        throw std::runtime_error("Free requires 'id' or 'handle'");
      }
    } else {
      throw std::runtime_error("Unsupported op: " + ev.op);
    }
    events.push_back(ev);
  }
  return events;
}

std::string ReplayStrategy(const std::string& strategy_name, uint64_t capacity,
                           uint64_t min_block, const std::vector<TraceEvent>& events,
                           bool strict_free) {
  slab_arena::SlabArena arena(capacity, ParseStrategy(strategy_name), min_block);
  std::unordered_map<std::string, uint64_t> live_ids;
  std::unordered_map<std::string, uint64_t> retired_ids;
  std::ostringstream ev_json;
  ev_json << "[";

  for (size_t i = 0; i < events.size(); ++i) {
    const TraceEvent& ev = events[i];
    if (i > 0) ev_json << ",";
    if (ev.op == "alloc" || ev.op == "aligned_alloc") {
      if (!ev.id.empty() && live_ids.count(ev.id)) {
        throw std::runtime_error("Duplicate live id: " + ev.id);
      }
      auto alloc = arena.Allocate(ev.size, ev.align);
      if (alloc.has_value()) {
        if (!ev.id.empty()) {
          live_ids[ev.id] = alloc->handle;
          retired_ids.erase(ev.id);
        }
        ev_json << "{\"op\":\"" << EscapeJson(ev.op) << "\",\"status\":\"accepted\""
                << ",\"handle\":" << alloc->handle
                << ",\"payload_offset\":" << alloc->payload_offset
                << ",\"raw_offset\":" << alloc->raw_offset
                << ",\"reserved_bytes\":" << alloc->reserved_size
                << ",\"alignment_padding\":" << alloc->alignment_padding << "}";
      } else {
        ev_json << "{\"op\":\"" << EscapeJson(ev.op) << "\",\"status\":\"failed\"}";
      }
    } else {
      uint64_t handle = 0;
      if (ev.has_raw_handle) handle = ev.raw_handle;
      else if (live_ids.count(ev.id)) handle = live_ids[ev.id];
      else if (retired_ids.count(ev.id)) handle = retired_ids[ev.id];
      const auto st = arena.Deallocate(handle);
      std::string status = "ok";
      if (st == slab_arena::FreeStatus::kOk) {
        if (!ev.id.empty() && live_ids.count(ev.id)) {
          retired_ids[ev.id] = live_ids[ev.id];
          live_ids.erase(ev.id);
        }
      } else {
        status = (st == slab_arena::FreeStatus::kDoubleFree) ? "double_free" : "invalid_handle";
        if (strict_free) throw std::runtime_error("Rejected free (" + status + ")");
      }
      ev_json << "{\"op\":\"free\",\"status\":\"" << status << "\"}";
    }
    std::string inv_err;
    if (!arena.VerifyInvariants(&inv_err)) {
      throw std::runtime_error("Invariant violation: " + inv_err);
    }
  }
  ev_json << "]";

  const auto s = arena.Stats();
  std::ostringstream out;
  out << std::fixed << std::setprecision(6)
      << "{\n  \"strategy\": \"" << EscapeJson(strategy_name) << "\",\n"
      << "  \"capacity_bytes\": " << s.capacity_bytes << ",\n"
      << "  \"accepted\": " << s.accepted << ",\n"
      << "  \"failed\": " << s.failed << ",\n"
      << "  \"freed\": " << s.freed << ",\n"
      << "  \"invalid_frees\": " << s.invalid_frees << ",\n"
      << "  \"double_frees\": " << s.double_frees << ",\n"
      << "  \"live_allocations\": " << s.live_allocations << ",\n"
      << "  \"live_reserved_bytes\": " << s.live_reserved_bytes << ",\n"
      << "  \"live_payload_bytes\": " << s.live_payload_bytes << ",\n"
      << "  \"total_free_bytes\": " << s.total_free_bytes << ",\n"
      << "  \"largest_free_bytes\": " << s.largest_free_bytes << ",\n"
      << "  \"peak_reserved_bytes\": " << s.peak_reserved_bytes << ",\n"
      << "  \"peak_payload_bytes\": " << s.peak_payload_bytes << ",\n"
      << "  \"alignment_overhead_bytes\": " << s.alignment_overhead_bytes << ",\n"
      << "  \"peak_alignment_overhead_bytes\": " << s.peak_alignment_overhead_bytes << ",\n"
      << "  \"external_fragmentation\": " << s.external_fragmentation << ",\n"
      << "  \"peak_external_fragmentation\": " << s.peak_external_fragmentation << ",\n"
      << "  \"invariants_verified\": true,\n"
      << "  \"events\": " << ev_json.str() << "\n}";
  return out.str();
}

void WriteExclusiveFile(const std::string& path, const std::string& content) {
  int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
  if (fd < 0) throw std::runtime_error("Cannot create '" + path + "': " + std::strerror(errno));
  size_t off = 0;
  while (off < content.size()) {
    ssize_t n = ::write(fd, content.data() + off, content.size() - off);
    if (n < 0) {
      int e = errno;
      ::close(fd);
      throw std::runtime_error(std::strerror(e));
    }
    off += static_cast<size_t>(n);
  }
  ::close(fd);
}

uint64_t ParseCliUint64(const std::string& s, const std::string& flag) {
  if (s.empty()) throw std::runtime_error("Empty integer for " + flag);
  uint64_t v = 0;
  for (char c : s) {
    if (!std::isdigit(static_cast<unsigned char>(c))) throw std::runtime_error("Bad int: " + s);
    uint64_t d = static_cast<uint64_t>(c - '0');
    if (v > (std::numeric_limits<uint64_t>::max() - d) / 10) throw std::runtime_error("Overflow");
    v = v * 10 + d;
  }
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string trace_path, strategy = "freelist_best_fit", output_path;
    uint64_t capacity = 65536, min_block = 16;
    bool strict_free = false;
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      auto next = [&]() -> std::string {
        if (i + 1 >= argc) throw std::runtime_error("Missing value for " + a);
        return argv[++i];
      };
      if (a == "--trace") trace_path = next();
      else if (a == "--strategy") strategy = next();
      else if (a == "--capacity") capacity = ParseCliUint64(next(), a);
      else if (a == "--min-block") min_block = ParseCliUint64(next(), a);
      else if (a == "--output") output_path = next();
      else if (a == "--strict-free") strict_free = true;
      else throw std::runtime_error("Unknown option: " + a);
    }
    if (trace_path.empty()) throw std::runtime_error("--trace <jsonl> required");
    const auto events = LoadTrace(trace_path);
    std::string rendered;
    if (strategy == "all") {
      std::ostringstream oss;
      oss << "{\n  \"strategies\": {\n";
      const char* names[3] = {"freelist_first_fit", "freelist_best_fit", "buddy"};
      for (int i = 0; i < 3; ++i) {
        if (i > 0) oss << ",\n";
        oss << "    \"" << names[i] << "\": "
            << ReplayStrategy(names[i], capacity, min_block, events, strict_free);
      }
      oss << "\n  }\n}\n";
      rendered = oss.str();
    } else {
      rendered = ReplayStrategy(strategy, capacity, min_block, events, strict_free) + "\n";
    }
    if (output_path.empty()) std::cout << rendered;
    else WriteExclusiveFile(output_path, rendered);
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "arena_verifier error: " << ex.what() << "\n";
    return 1;
  }
}
