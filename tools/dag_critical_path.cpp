#include "task_dag.hpp"

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
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

struct JVal {
  using Obj = std::map<std::string, JVal>;
  using Arr = std::vector<JVal>;
  std::variant<std::nullptr_t, bool, double, std::string, Arr, Obj> v;
  const Obj& AsObj() const { return std::get<Obj>(v); }
  const Arr& AsArr() const { return std::get<Arr>(v); }
  const std::string& AsStr() const { return std::get<std::string>(v); }
  double AsNum() const { return std::get<double>(v); }
};

class MiniJsonParser {
 public:
  explicit MiniJsonParser(const std::string& src) : s_(src) {}
  JVal Parse() {
    SkipWs();
    JVal r = ParseValue();
    SkipWs();
    if (p_ != s_.size()) throw std::runtime_error("Trailing JSON characters");
    return r;
  }

 private:
  void SkipWs() {
    while (p_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[p_]))) ++p_;
  }
  void Expect(char c) {
    if (p_ >= s_.size() || s_[p_] != c) throw std::runtime_error("Expected JSON delimiter");
    ++p_;
  }
  JVal ParseValue() {
    SkipWs();
    if (p_ >= s_.size()) throw std::runtime_error("Unexpected EOF in JSON");
    char c = s_[p_];
    if (c == '{') return ParseObject();
    if (c == '[') return ParseArray();
    if (c == '"') return JVal{ParseString()};
    if (s_.compare(p_, 4, "true") == 0) { p_ += 4; return JVal{true}; }
    if (s_.compare(p_, 5, "false") == 0) { p_ += 5; return JVal{false}; }
    if (s_.compare(p_, 4, "null") == 0) { p_ += 4; return JVal{nullptr}; }
    return JVal{ParseNumber()};
  }
  JVal ParseObject() {
    Expect('{');
    JVal::Obj m;
    SkipWs();
    if (p_ < s_.size() && s_[p_] == '}') { ++p_; return JVal{std::move(m)}; }
    while (true) {
      SkipWs();
      std::string k = ParseString();
      SkipWs();
      Expect(':');
      m[k] = ParseValue();
      SkipWs();
      if (p_ < s_.size() && s_[p_] == '}') { ++p_; break; }
      Expect(',');
    }
    return JVal{std::move(m)};
  }
  JVal ParseArray() {
    Expect('[');
    JVal::Arr a;
    SkipWs();
    if (p_ < s_.size() && s_[p_] == ']') { ++p_; return JVal{std::move(a)}; }
    while (true) {
      a.push_back(ParseValue());
      SkipWs();
      if (p_ < s_.size() && s_[p_] == ']') { ++p_; break; }
      Expect(',');
    }
    return JVal{std::move(a)};
  }
  std::string ParseString() {
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
    throw std::runtime_error("Unterminated JSON string");
  }
  double ParseNumber() {
    size_t b = p_;
    if (s_[p_] == '-') ++p_;
    while (p_ < s_.size()) {
      char c = s_[p_];
      if (std::isdigit(static_cast<unsigned char>(c)) || c == '.' || c == 'e' ||
          c == 'E' || c == '+' || c == '-') {
        ++p_;
      } else {
        break;
      }
    }
    std::string tok = s_.substr(b, p_ - b);
    if (tok.empty() || tok == "-") throw std::runtime_error("Invalid number");
    size_t consumed = 0;
    double val = std::stod(tok, &consumed);
    if (consumed != tok.size() || !std::isfinite(val)) throw std::runtime_error("Bad number");
    return val;
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

void WriteExclusiveFile(const std::string& path, const std::string& content) {
  int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
  if (fd < 0) throw std::runtime_error("Cannot create '" + path + "': " + std::strerror(errno));
  size_t off = 0;
  while (off < content.size()) {
    ssize_t n = ::write(fd, content.data() + off, content.size() - off);
    if (n < 0) {
      int saved = errno;
      ::close(fd);
      throw std::runtime_error(std::strerror(saved));
    }
    off += static_cast<size_t>(n);
  }
  ::close(fd);
}

uint32_t ParseWorkers(const std::string& s) {
  if (s.empty()) throw std::runtime_error("Empty --workers value");
  uint64_t v = 0;
  for (char c : s) {
    if (!std::isdigit(static_cast<unsigned char>(c))) throw std::runtime_error("Bad --workers");
    v = v * 10 + static_cast<uint64_t>(c - '0');
    if (v > 65535) throw std::runtime_error("--workers exceeds 65535");
  }
  if (v == 0) throw std::runtime_error("--workers must be >= 1");
  return static_cast<uint32_t>(v);
}

std::vector<task_dag::TaskSpec> LoadDagSpecs(const std::string& path) {
  std::ifstream in(path);
  if (!in.is_open()) throw std::runtime_error("Cannot open DAG JSON: " + path);
  std::string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  const JVal root = MiniJsonParser(body).Parse();
  const auto& obj = root.AsObj();
  if (!obj.count("tasks")) throw std::runtime_error("DAG JSON missing 'tasks' array");

  std::vector<task_dag::TaskSpec> out;
  for (const JVal& tv : obj.at("tasks").AsArr()) {
    const auto& tobj = tv.AsObj();
    if (!tobj.count("id") || !tobj.count("cost")) {
      throw std::runtime_error("Task requires 'id' and 'cost'");
    }
    task_dag::TaskSpec spec{tobj.at("id").AsStr(), tobj.at("cost").AsNum(), {}};
    if (tobj.count("deps")) {
      for (const JVal& dv : tobj.at("deps").AsArr()) spec.deps.push_back(dv.AsStr());
    }
    out.push_back(std::move(spec));
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string input_path, output_path;
    uint32_t max_workers = 4;
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      auto next = [&]() -> std::string {
        if (i + 1 >= argc) throw std::runtime_error("Missing value for " + a);
        return argv[++i];
      };
      if (a == "--input") input_path = next();
      else if (a == "--workers") max_workers = ParseWorkers(next());
      else if (a == "--output") output_path = next();
      else throw std::runtime_error("Unknown option: " + a);
    }
    if (input_path.empty()) throw std::runtime_error("--input <dag.json> required");

    task_dag::TaskDag dag(LoadDagSpecs(input_path));
    const auto vr = dag.Validate();
    if (!vr.valid) {
      std::cerr << "dag_critical_path validation error [" << vr.code << "]: " << vr.message << "\n";
      return 2;
    }

    const auto cpm = dag.AnalyzeCpm();
    std::vector<task_dag::ScheduleResult> sims;
    for (uint32_t k = 1; k <= max_workers; ++k) {
      auto sr = dag.SimulateWorkers(k);
      if (!sr.bounds_satisfied) throw std::runtime_error("Makespan lower-bound violated");
      sims.push_back(std::move(sr));
    }

    std::ostringstream out;
    out << std::fixed << std::setprecision(6)
        << "{\n  \"valid\": true,\n  \"task_count\": " << cpm.nodes.size()
        << ",\n  \"total_work\": " << cpm.total_work
        << ",\n  \"critical_path_cost\": " << cpm.critical_path_cost
        << ",\n  \"topological_order\": [";
    for (size_t i = 0; i < cpm.topological_order.size(); ++i) {
      if (i > 0) out << ", ";
      out << "\"" << EscapeJson(cpm.topological_order[i]) << "\"";
    }
    out << "],\n  \"critical_path\": [";
    for (size_t i = 0; i < cpm.critical_path.size(); ++i) {
      if (i > 0) out << ", ";
      out << "\"" << EscapeJson(cpm.critical_path[i]) << "\"";
    }
    out << "],\n  \"slack_table\": [";
    for (size_t i = 0; i < cpm.nodes.size(); ++i) {
      if (i > 0) out << ", ";
      const auto& n = cpm.nodes[i];
      out << "{\"id\":\"" << EscapeJson(n.id) << "\",\"cost\":" << n.cost
          << ",\"earliest_start\":" << n.earliest_start
          << ",\"earliest_finish\":" << n.earliest_finish
          << ",\"latest_start\":" << n.latest_start
          << ",\"latest_finish\":" << n.latest_finish
          << ",\"total_slack\":" << n.total_slack
          << ",\"critical\":" << (n.critical ? "true" : "false") << "}";
    }
    out << "],\n  \"schedules\": [";
    for (size_t i = 0; i < sims.size(); ++i) {
      if (i > 0) out << ", ";
      const auto& s = sims[i];
      out << "{\"workers\":" << s.workers << ",\"makespan\":" << s.makespan
          << ",\"lower_bound\":" << s.lower_bound
          << ",\"bounds_satisfied\":" << (s.bounds_satisfied ? "true" : "false")
          << ",\"average_utilization\":" << s.average_utilization
          << ",\"speedup_vs_serial\":" << s.speedup_vs_serial << ",\"worker_utilization\":[";
      for (size_t w = 0; w < s.worker_utilization.size(); ++w) {
        if (w > 0) out << ",";
        out << s.worker_utilization[w];
      }
      out << "],\"timeline\":[";
      for (size_t t = 0; t < s.timeline.size(); ++t) {
        if (t > 0) out << ",";
        const auto& sl = s.timeline[t];
        out << "{\"task_id\":\"" << EscapeJson(sl.task_id) << "\",\"worker_id\":" << sl.worker_id
            << ",\"start_time\":" << sl.start_time << ",\"finish_time\":" << sl.finish_time << "}";
      }
      out << "]}";
    }
    out << "]\n}\n";

    if (output_path.empty()) std::cout << out.str();
    else WriteExclusiveFile(output_path, out.str());
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "dag_critical_path error: " << ex.what() << "\n";
    return 1;
  }
}
