#include "tensor_container.hpp"

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <variant>
#include <vector>

namespace {

struct JValue {
  using Obj = std::map<std::string, JValue>;
  using Arr = std::vector<JValue>;
  std::variant<std::nullptr_t, bool, int64_t, double, std::string, Arr, Obj> v;
  const Obj& AsObj() const { return std::get<Obj>(v); }
  const Arr& AsArr() const { return std::get<Arr>(v); }
  const std::string& AsStr() const { return std::get<std::string>(v); }
  int64_t AsInt() const { return std::get<int64_t>(v); }
  double AsNumber() const {
    return std::holds_alternative<int64_t>(v) ? static_cast<double>(std::get<int64_t>(v))
                                              : std::get<double>(v);
  }
};

class JsonParser {
 public:
  explicit JsonParser(const std::string& s) : s_(s) {}
  JValue Parse() {
    SkipWs();
    JValue r = ParseVal();
    SkipWs();
    if (p_ != s_.size()) throw std::runtime_error("Trailing tokens in JSON");
    return r;
  }

 private:
  void SkipWs() {
    while (p_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[p_]))) ++p_;
  }
  void Expect(char c) {
    if (p_ >= s_.size() || s_[p_] != c) throw std::runtime_error("Unexpected JSON char");
    ++p_;
  }
  JValue ParseVal() {
    SkipWs();
    if (p_ >= s_.size()) throw std::runtime_error("Unexpected EOF in JSON");
    char c = s_[p_];
    if (c == '{') return ParseObj();
    if (c == '[') return ParseArr();
    if (c == '"') return JValue{ParseStr()};
    if (s_.compare(p_, 4, "true") == 0) { p_ += 4; return JValue{true}; }
    if (s_.compare(p_, 5, "false") == 0) { p_ += 5; return JValue{false}; }
    if (s_.compare(p_, 4, "null") == 0) { p_ += 4; return JValue{nullptr}; }
    return ParseNum();
  }
  JValue ParseObj() {
    Expect('{');
    JValue::Obj obj;
    SkipWs();
    if (p_ < s_.size() && s_[p_] == '}') { ++p_; return JValue{std::move(obj)}; }
    while (true) {
      SkipWs();
      std::string k = ParseStr();
      SkipWs();
      Expect(':');
      obj[k] = ParseVal();
      SkipWs();
      if (p_ < s_.size() && s_[p_] == '}') { ++p_; break; }
      Expect(',');
    }
    return JValue{std::move(obj)};
  }
  JValue ParseArr() {
    Expect('[');
    JValue::Arr arr;
    SkipWs();
    if (p_ < s_.size() && s_[p_] == ']') { ++p_; return JValue{std::move(arr)}; }
    while (true) {
      arr.push_back(ParseVal());
      SkipWs();
      if (p_ < s_.size() && s_[p_] == ']') { ++p_; break; }
      Expect(',');
    }
    return JValue{std::move(arr)};
  }
  std::string ParseStr() {
    Expect('"');
    std::string out;
    while (p_ < s_.size()) {
      char c = s_[p_++];
      if (c == '"') return out;
      if (c == '\\') {
        if (p_ >= s_.size()) throw std::runtime_error("Bad escape");
        char e = s_[p_++];
        if (e == '"' || e == '\\' || e == '/') out.push_back(e);
        else if (e == 'n') out.push_back('\n');
        else if (e == 'r') out.push_back('\r');
        else if (e == 't') out.push_back('\t');
        else throw std::runtime_error("Unsupported escape");
      } else {
        out.push_back(c);
      }
    }
    throw std::runtime_error("Unterminated string");
  }
  JValue ParseNum() {
    size_t start = p_;
    if (s_[p_] == '-') ++p_;
    bool is_float = false;
    while (p_ < s_.size()) {
      char c = s_[p_];
      if (std::isdigit(static_cast<unsigned char>(c))) ++p_;
      else if (c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') { is_float = true; ++p_; }
      else break;
    }
    std::string tok = s_.substr(start, p_ - start);
    if (tok.empty() || tok == "-") throw std::runtime_error("Invalid number");
    size_t idx = 0;
    if (is_float) {
      double dv = std::stod(tok, &idx);
      if (idx != tok.size() || !std::isfinite(dv)) throw std::runtime_error("Bad float");
      return JValue{dv};
    }
    long long iv = std::stoll(tok, &idx, 10);
    if (idx != tok.size()) throw std::runtime_error("Bad int");
    return JValue{static_cast<int64_t>(iv)};
  }

  const std::string& s_;
  size_t p_ = 0;
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

void WriteExclusiveBytes(const std::string& path, const uint8_t* data, size_t len) {
  int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
  if (fd < 0) throw std::runtime_error("Cannot create '" + path + "': " + std::strerror(errno));
  size_t off = 0;
  while (off < len) {
    ssize_t n = ::write(fd, data + off, len - off);
    if (n < 0) {
      int e = errno;
      ::close(fd);
      throw std::runtime_error(std::strerror(e));
    }
    off += static_cast<size_t>(n);
  }
  ::close(fd);
}

std::vector<uint8_t> ReadBinaryFile(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary | std::ios::ate);
  if (!in.is_open()) throw std::runtime_error("Cannot open file: " + p.string());
  const std::streamsize sz = in.tellg();
  std::vector<uint8_t> buf(static_cast<size_t>(sz));
  in.seekg(0, std::ios::beg);
  if (sz > 0 && !in.read(reinterpret_cast<char*>(buf.data()), sz)) {
    throw std::runtime_error("Read failed: " + p.string());
  }
  return buf;
}

std::string RenderReportJson(const tensor_container::AuditReport& rep, bool include_audit) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(6) << "{\n";
  if (include_audit) out << "  \"valid\": " << (rep.valid ? "true" : "false") << ",\n";
  out << "  \"file_bytes\": " << rep.file_bytes << ",\n"
      << "  \"payload_section_offset\": " << rep.payload_section_offset << ",\n"
      << "  \"metadata\": [";
  for (size_t i = 0; i < rep.metadata.size(); ++i) {
    if (i > 0) out << ", ";
    const auto& m = rep.metadata[i];
    out << "{\"key\":\"" << EscapeJson(m.key) << "\",\"type\":\"";
    if (m.type == tensor_container::MetaType::kString) {
      out << "string\",\"value\":\"" << EscapeJson(std::get<std::string>(m.value)) << "\"}";
    } else if (m.type == tensor_container::MetaType::kInt64) {
      out << "int64\",\"value\":" << std::get<int64_t>(m.value) << "}";
    } else if (m.type == tensor_container::MetaType::kFloat64) {
      out << "float64\",\"value\":" << std::get<double>(m.value) << "}";
    } else {
      out << "bool\",\"value\":" << (std::get<bool>(m.value) ? "true" : "false") << "}";
    }
  }
  out << "],\n  \"tensors\": [";
  for (size_t i = 0; i < rep.tensors.size(); ++i) {
    if (i > 0) out << ", ";
    const auto& t = rep.tensors[i];
    out << "{\"name\":\"" << EscapeJson(t.name) << "\",\"dtype\":\""
        << tensor_container::DTypeName(t.dtype) << "\",\"rank\":" << t.dims.size() << ",\"dims\":[";
    for (size_t d = 0; d < t.dims.size(); ++d) {
      if (d > 0) out << ",";
      out << t.dims[d];
    }
    out << "],\"alignment\":" << t.alignment << ",\"payload_offset\":" << t.payload_offset
        << ",\"payload_size\":" << t.payload_size << ",\"crc32\":" << t.crc32 << "}";
  }
  out << "]";
  if (include_audit) {
    out << ",\n  \"findings\": [";
    for (size_t i = 0; i < rep.findings.size(); ++i) {
      if (i > 0) out << ", ";
      out << "{\"code\":\"" << EscapeJson(rep.findings[i].code) << "\",\"message\":\""
          << EscapeJson(rep.findings[i].message) << "\"}";
    }
    out << "]";
  }
  out << "\n}\n";
  return out.str();
}

int RunPack(const std::string& manifest_path, const std::string& output_path) {
  std::ifstream in(manifest_path);
  if (!in.is_open()) throw std::runtime_error("Cannot open manifest: " + manifest_path);
  std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  const JValue root = JsonParser(text).Parse();
  const auto& obj = root.AsObj();
  const auto base_dir = std::filesystem::path(manifest_path).parent_path();

  std::vector<tensor_container::MetadataEntry> meta;
  if (obj.count("metadata")) {
    for (const JValue& mv : obj.at("metadata").AsArr()) {
      const auto& mobj = mv.AsObj();
      tensor_container::MetadataEntry e;
      e.key = mobj.at("key").AsStr();
      const std::string tn = mobj.at("type").AsStr();
      const JValue& val = mobj.at("value");
      if (tn == "string") { e.type = tensor_container::MetaType::kString; e.value = val.AsStr(); }
      else if (tn == "int64") { e.type = tensor_container::MetaType::kInt64; e.value = val.AsInt(); }
      else if (tn == "float64") { e.type = tensor_container::MetaType::kFloat64; e.value = val.AsNumber(); }
      else if (tn == "bool") { e.type = tensor_container::MetaType::kBool; e.value = std::get<bool>(val.v); }
      else throw std::runtime_error("Unsupported metadata type: " + tn);
      meta.push_back(std::move(e));
    }
  }

  std::vector<tensor_container::TensorInput> tensors;
  for (const JValue& tv : obj.at("tensors").AsArr()) {
    const auto& tobj = tv.AsObj();
    tensor_container::TensorInput t;
    t.name = tobj.at("name").AsStr();
    auto dt = tensor_container::ParseDType(tobj.at("dtype").AsStr());
    if (!dt.has_value()) throw std::runtime_error("Unknown dtype");
    t.dtype = *dt;
    for (const JValue& dv : tobj.at("dims").AsArr()) {
      if (dv.AsInt() < 0) throw std::runtime_error("Negative tensor dim");
      t.dims.push_back(static_cast<uint64_t>(dv.AsInt()));
    }
    t.alignment = tobj.count("alignment") ? static_cast<uint64_t>(tobj.at("alignment").AsInt()) : 64ULL;
    std::filesystem::path bp = tobj.at("path").AsStr();
    if (bp.is_relative()) bp = base_dir / bp;
    t.data = ReadBinaryFile(bp);
    tensors.push_back(std::move(t));
  }
  const auto archive = tensor_container::BuildArchive(meta, tensors);
  WriteExclusiveBytes(output_path, archive.data(), archive.size());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) throw std::runtime_error("Usage: container_audit <pack|inspect|audit> ...");
    const std::string subcmd = argv[1];
    std::string manifest_path, input_path, output_path;
    for (int i = 2; i < argc; ++i) {
      std::string a = argv[i];
      auto next = [&]() -> std::string {
        if (i + 1 >= argc) throw std::runtime_error("Missing value for " + a);
        return argv[++i];
      };
      if (a == "--manifest") manifest_path = next();
      else if (a == "--input") input_path = next();
      else if (a == "--output") output_path = next();
      else throw std::runtime_error("Unknown flag: " + a);
    }
    if (subcmd == "pack") {
      if (manifest_path.empty() || output_path.empty()) throw std::runtime_error("Missing flags");
      return RunPack(manifest_path, output_path);
    }
    if (subcmd == "inspect" || subcmd == "audit") {
      if (input_path.empty()) throw std::runtime_error("--input required");
      const auto rep = tensor_container::AuditArchiveFile(input_path);
      if (subcmd == "inspect" && !rep.valid) {
        throw std::runtime_error("Container failed validation: " + rep.findings.front().code);
      }
      const std::string js = RenderReportJson(rep, subcmd == "audit");
      if (output_path.empty()) std::cout << js;
      else WriteExclusiveBytes(output_path, reinterpret_cast<const uint8_t*>(js.data()), js.size());
      return (subcmd == "audit" && !rep.valid) ? 2 : 0;
    }
    throw std::runtime_error("Unknown subcommand: " + subcmd);
  } catch (const std::exception& ex) {
    std::cerr << "container_audit error: " << ex.what() << "\n";
    return 1;
  }
}
