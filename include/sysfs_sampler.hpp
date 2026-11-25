#ifndef WAVE_SYSFS_SAMPLER_HPP_
#define WAVE_SYSFS_SAMPLER_HPP_

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sysfs_sampler {

struct CpuJiffies {
  std::string label;
  int cpu_index = -1;
  uint64_t user = 0, nice = 0, system = 0, idle = 0, iowait = 0, irq = 0, softirq = 0, steal = 0;
  uint64_t IdleAll() const noexcept { return idle + iowait; }
  uint64_t BusyAll() const noexcept { return user + nice + system + irq + softirq + steal; }
  uint64_t Total() const noexcept { return IdleAll() + BusyAll(); }
};

struct CpuUtilizationDelta {
  std::string label;
  int cpu_index = -1;
  uint64_t delta_total = 0, delta_busy = 0, delta_idle = 0, delta_iowait = 0;
  double busy_fraction = 0.0, user_fraction = 0.0, system_fraction = 0.0;
  double iowait_fraction = 0.0, irq_fraction = 0.0;
};

struct ProcessStatus {
  int32_t pid = 0;
  std::string name;
  std::string state;
  uint64_t vm_rss_kb = 0;
  uint64_t vm_peak_kb = 0;
  uint32_t threads = 0;
  uint64_t voluntary_ctxt_switches = 0;
  uint64_t nonvoluntary_ctxt_switches = 0;
};

struct PsiWindow {
  bool present = false;
  double avg10 = 0.0, avg60 = 0.0, avg300 = 0.0;
  uint64_t total_us = 0;
};

struct PsiResource {
  PsiWindow some;
  PsiWindow full;
};

struct ThermalZoneSample {
  int zone_id = 0;
  std::string type;
  int64_t temp_millideg_c = 0;
  double temp_celsius = 0.0;
};

struct CpuFreqSample {
  int cpu_id = 0;
  uint64_t scaling_cur_freq_khz = 0;
  uint64_t scaling_max_freq_khz = 0;
  double utilization_of_max = 0.0;
};

struct TelemetrySnapshot {
  std::map<std::string, uint64_t> meminfo_kb;
  CpuJiffies total_cpu;
  std::vector<CpuJiffies> per_cpu;
  std::optional<ProcessStatus> process;
  std::map<std::string, PsiResource> pressure;
  std::vector<ThermalZoneSample> thermal_zones;
  std::vector<CpuFreqSample> cpufreq;
};

struct TelemetryDelta {
  CpuUtilizationDelta total_cpu;
  std::vector<CpuUtilizationDelta> per_cpu;
  std::optional<uint64_t> delta_voluntary_ctxt_switches;
  std::optional<uint64_t> delta_nonvoluntary_ctxt_switches;
  std::map<std::string, uint64_t> psi_some_delta_us;
  std::map<std::string, uint64_t> psi_full_delta_us;
};

namespace detail {

inline std::string Trim(const std::string& s) {
  size_t b = 0, e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

inline bool AllDigits(const std::string& s) noexcept {
  if (s.empty()) return false;
  for (char c : s) {
    if (!std::isdigit(static_cast<unsigned char>(c))) return false;
  }
  return true;
}

inline uint64_t ParseStrictUint64(const std::string& raw, const std::string& ctx) {
  const std::string s = Trim(raw);
  if (!AllDigits(s)) throw std::runtime_error("Invalid uint64 in " + ctx + ": '" + s + "'");
  uint64_t val = 0;
  for (char c : s) {
    const uint64_t d = static_cast<uint64_t>(c - '0');
    if (val > (std::numeric_limits<uint64_t>::max() - d) / 10) {
      throw std::runtime_error("Uint64 overflow in " + ctx);
    }
    val = val * 10 + d;
  }
  return val;
}

inline int64_t ParseStrictInt64(const std::string& raw, const std::string& ctx) {
  const std::string s = Trim(raw);
  if (s.empty()) throw std::runtime_error("Empty int64 in " + ctx);
  bool neg = (s[0] == '-');
  size_t pos = (s[0] == '-' || s[0] == '+') ? 1 : 0;
  const std::string digits = s.substr(pos);
  if (!AllDigits(digits)) throw std::runtime_error("Invalid int64 in " + ctx + ": '" + s + "'");
  uint64_t mag = ParseStrictUint64(digits, ctx);
  if (!neg) {
    if (mag > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      throw std::runtime_error("Int64 overflow in " + ctx);
    }
    return static_cast<int64_t>(mag);
  }
  const uint64_t max_neg = static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + 1ULL;
  if (mag > max_neg) throw std::runtime_error("Int64 underflow in " + ctx);
  return (mag == max_neg) ? std::numeric_limits<int64_t>::min() : -static_cast<int64_t>(mag);
}

inline double ParseStrictDouble(const std::string& raw, const std::string& ctx) {
  const std::string s = Trim(raw);
  if (s.empty()) throw std::runtime_error("Empty float in " + ctx);
  size_t i = (s[0] == '+' || s[0] == '-') ? 1 : 0;
  bool saw_digit = false, saw_dot = false;
  for (; i < s.size(); ++i) {
    if (std::isdigit(static_cast<unsigned char>(s[i]))) saw_digit = true;
    else if (s[i] == '.' && !saw_dot) saw_dot = true;
    else throw std::runtime_error("Invalid float in " + ctx + ": '" + s + "'");
  }
  if (!saw_digit) throw std::runtime_error("Float lacks digits in " + ctx);
  size_t consumed = 0;
  double v = std::stod(s, &consumed);
  if (consumed != s.size() || !std::isfinite(v)) throw std::runtime_error("Bad float in " + ctx);
  return v;
}

inline std::string ReadSingleLineFile(const std::filesystem::path& p) {
  std::ifstream in(p);
  if (!in.is_open()) throw std::runtime_error("Cannot open sysfs node: " + p.string());
  std::string line;
  if (!std::getline(in, line)) throw std::runtime_error("Empty sysfs node: " + p.string());
  return Trim(line);
}

}  // namespace detail

class SysfsSampler {
 public:
  explicit SysfsSampler(std::filesystem::path sysroot = "/") : sysroot_(std::move(sysroot)) {}

  TelemetrySnapshot Sample(std::optional<int32_t> pid = std::nullopt) const {
    if (!std::filesystem::exists(sysroot_)) {
      throw std::runtime_error("Sysroot does not exist: " + sysroot_.string());
    }
    TelemetrySnapshot snap;
    ParseMeminfo(&snap.meminfo_kb);
    ParseProcStat(&snap.total_cpu, &snap.per_cpu);
    if (pid.has_value()) snap.process = ParsePidStatus(*pid);
    ParsePressure(&snap.pressure);
    ParseThermal(&snap.thermal_zones);
    ParseCpuFreq(&snap.cpufreq);
    return snap;
  }

  static CpuUtilizationDelta ComputeCpuDelta(const CpuJiffies& a, const CpuJiffies& b) {
    if (a.label != b.label) throw std::runtime_error("Mismatched CPU label in delta");
    auto sub = [](uint64_t p, uint64_t c) noexcept { return (c >= p) ? (c - p) : 0ULL; };
    const uint64_t du = sub(a.user, b.user) + sub(a.nice, b.nice);
    const uint64_t ds = sub(a.system, b.system);
    const uint64_t di = sub(a.idle, b.idle);
    const uint64_t dw = sub(a.iowait, b.iowait);
    const uint64_t dq = sub(a.irq, b.irq) + sub(a.softirq, b.softirq);
    const uint64_t dst = sub(a.steal, b.steal);
    const uint64_t db = du + ds + dq + dst;
    const uint64_t dt = db + di + dw;

    CpuUtilizationDelta out{b.label, b.cpu_index, dt, db, di, dw, 0.0, 0.0, 0.0, 0.0, 0.0};
    if (dt > 0) {
      const double denom = static_cast<double>(dt);
      out.busy_fraction = static_cast<double>(db) / denom;
      out.user_fraction = static_cast<double>(du) / denom;
      out.system_fraction = static_cast<double>(ds) / denom;
      out.iowait_fraction = static_cast<double>(dw) / denom;
      out.irq_fraction = static_cast<double>(dq) / denom;
    }
    return out;
  }

  static TelemetryDelta ComputeDelta(const TelemetrySnapshot& a, const TelemetrySnapshot& b) {
    TelemetryDelta d;
    d.total_cpu = ComputeCpuDelta(a.total_cpu, b.total_cpu);
    if (a.per_cpu.size() != b.per_cpu.size()) {
      throw std::runtime_error("Per-CPU count changed across snapshots");
    }
    for (size_t i = 0; i < b.per_cpu.size(); ++i) {
      d.per_cpu.push_back(ComputeCpuDelta(a.per_cpu[i], b.per_cpu[i]));
    }
    if (a.process.has_value() && b.process.has_value() && a.process->pid == b.process->pid) {
      d.delta_voluntary_ctxt_switches =
          (b.process->voluntary_ctxt_switches >= a.process->voluntary_ctxt_switches)
              ? (b.process->voluntary_ctxt_switches - a.process->voluntary_ctxt_switches)
              : 0ULL;
      d.delta_nonvoluntary_ctxt_switches =
          (b.process->nonvoluntary_ctxt_switches >= a.process->nonvoluntary_ctxt_switches)
              ? (b.process->nonvoluntary_ctxt_switches - a.process->nonvoluntary_ctxt_switches)
              : 0ULL;
    }
    for (const auto& [res, b_psi] : b.pressure) {
      auto it = a.pressure.find(res);
      if (it == a.pressure.end()) continue;
      if (it->second.some.present && b_psi.some.present &&
          b_psi.some.total_us >= it->second.some.total_us) {
        d.psi_some_delta_us[res] = b_psi.some.total_us - it->second.some.total_us;
      }
      if (it->second.full.present && b_psi.full.present &&
          b_psi.full.total_us >= it->second.full.total_us) {
        d.psi_full_delta_us[res] = b_psi.full.total_us - it->second.full.total_us;
      }
    }
    return d;
  }

 private:
  void ParseMeminfo(std::map<std::string, uint64_t>* out) const {
    std::ifstream in(sysroot_ / "proc" / "meminfo");
    if (!in.is_open()) throw std::runtime_error("Cannot open proc/meminfo");
    std::string line;
    while (std::getline(in, line)) {
      const std::string t = detail::Trim(line);
      if (t.empty()) continue;
      const size_t colon = t.find(':');
      if (colon == std::string::npos || colon == 0) throw std::runtime_error("Bad meminfo row");
      const std::string key = detail::Trim(t.substr(0, colon));
      std::istringstream rhs(t.substr(colon + 1));
      std::string num_tok, unit_tok, extra;
      if (!(rhs >> num_tok)) throw std::runtime_error("Missing meminfo value for " + key);
      const uint64_t val = detail::ParseStrictUint64(num_tok, "meminfo:" + key);
      if (rhs >> unit_tok) {
        if (unit_tok != "kB" || (rhs >> extra)) {
          throw std::runtime_error("Invalid meminfo unit for " + key);
        }
      }
      (*out)[key] = val;
    }
    if (!out->count("MemTotal")) throw std::runtime_error("MemTotal missing in meminfo");
  }

  void ParseProcStat(CpuJiffies* total_out, std::vector<CpuJiffies>* per_cpu_out) const {
    std::ifstream in(sysroot_ / "proc" / "stat");
    if (!in.is_open()) throw std::runtime_error("Cannot open proc/stat");
    bool saw_total = false;
    std::string line;
    while (std::getline(in, line)) {
      const std::string t = detail::Trim(line);
      if (t.empty()) continue;
      std::istringstream iss(t);
      std::string tag;
      iss >> tag;
      if (tag.rfind("cpu", 0) != 0) continue;
      if (tag != "cpu" && !detail::AllDigits(tag.substr(3))) continue;
      std::vector<uint64_t> cols;
      std::string tok;
      while (iss >> tok) cols.push_back(detail::ParseStrictUint64(tok, "stat:" + tag));
      if (cols.size() < 7) throw std::runtime_error("Short cpu row in proc/stat");
      CpuJiffies j{tag,
                   (tag == "cpu") ? -1 : static_cast<int>(detail::ParseStrictUint64(tag.substr(3), "cpu")),
                   cols[0], cols[1], cols[2], cols[3], cols[4], cols[5], cols[6],
                   (cols.size() >= 8) ? cols[7] : 0};
      if (tag == "cpu") {
        *total_out = j;
        saw_total = true;
      } else {
        per_cpu_out->push_back(j);
      }
    }
    if (!saw_total) throw std::runtime_error("Aggregate cpu row missing in proc/stat");
    std::sort(per_cpu_out->begin(), per_cpu_out->end(),
              [](const CpuJiffies& x, const CpuJiffies& y) { return x.cpu_index < y.cpu_index; });
  }

  ProcessStatus ParsePidStatus(int32_t pid) const {
    if (pid <= 0) throw std::runtime_error("PID must be positive");
    std::ifstream in(sysroot_ / "proc" / std::to_string(pid) / "status");
    if (!in.is_open()) throw std::runtime_error("Cannot open proc/<pid>/status");
    ProcessStatus st;
    st.pid = pid;
    bool saw_threads = false;
    std::string line;
    while (std::getline(in, line)) {
      const size_t colon = line.find(':');
      if (colon == std::string::npos) continue;
      const std::string k = detail::Trim(line.substr(0, colon));
      const std::string v = detail::Trim(line.substr(colon + 1));
      auto parse_kb = [&](const std::string& ctx) -> uint64_t {
        std::istringstream iss(v);
        std::string n, u;
        if (!(iss >> n) || !(iss >> u) || u != "kB") throw std::runtime_error("Bad kB in " + ctx);
        return detail::ParseStrictUint64(n, ctx);
      };
      if (k == "Name") st.name = v;
      else if (k == "State") st.state = v;
      else if (k == "VmRSS") st.vm_rss_kb = parse_kb("VmRSS");
      else if (k == "VmPeak") st.vm_peak_kb = parse_kb("VmPeak");
      else if (k == "Threads") {
        st.threads = static_cast<uint32_t>(detail::ParseStrictUint64(v, "Threads"));
        saw_threads = true;
      } else if (k == "voluntary_ctxt_switches") {
        st.voluntary_ctxt_switches = detail::ParseStrictUint64(v, k);
      } else if (k == "nonvoluntary_ctxt_switches") {
        st.nonvoluntary_ctxt_switches = detail::ParseStrictUint64(v, k);
      }
    }
    if (!saw_threads) throw std::runtime_error("Missing Threads in status");
    return st;
  }

  void ParsePressure(std::map<std::string, PsiResource>* out) const {
    const auto dir = sysroot_ / "proc" / "pressure";
    if (!std::filesystem::exists(dir)) return;
    constexpr std::array<const char*, 3> kKinds = {"cpu", "memory", "io"};
    for (const char* name : kKinds) {
      const auto p = dir / name;
      if (!std::filesystem::exists(p)) continue;
      std::ifstream in(p);
      if (!in.is_open()) throw std::runtime_error("Cannot open PSI node");
      PsiResource res;
      std::string line;
      while (std::getline(in, line)) {
        const std::string t = detail::Trim(line);
        if (t.empty()) continue;
        std::istringstream iss(t);
        std::string kind, kv;
        iss >> kind;
        PsiWindow* w = (kind == "some") ? &res.some : ((kind == "full") ? &res.full : nullptr);
        if (w == nullptr) throw std::runtime_error("Invalid PSI kind: " + kind);
        w->present = true;
        int mask = 0;
        while (iss >> kv) {
          const size_t eq = kv.find('=');
          if (eq == std::string::npos) throw std::runtime_error("Malformed PSI token");
          const std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
          auto pct = [&](const std::string& ctx) {
            double x = detail::ParseStrictDouble(v, ctx);
            if (x < 0.0 || x > 100.0) throw std::runtime_error("PSI pct out of bounds");
            return x;
          };
          if (k == "avg10") { w->avg10 = pct("avg10"); mask |= 1; }
          else if (k == "avg60") { w->avg60 = pct("avg60"); mask |= 2; }
          else if (k == "avg300") { w->avg300 = pct("avg300"); mask |= 4; }
          else if (k == "total") { w->total_us = detail::ParseStrictUint64(v, "total"); mask |= 8; }
          else throw std::runtime_error("Unknown PSI key: " + k);
        }
        if ((mask & 11) != 11) throw std::runtime_error("Incomplete PSI fields");
      }
      if (!res.some.present) throw std::runtime_error("PSI missing 'some'");
      (*out)[name] = res;
    }
  }

  void ParseThermal(std::vector<ThermalZoneSample>* out) const {
    const auto dir = sysroot_ / "sys" / "class" / "thermal";
    if (!std::filesystem::exists(dir)) return;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
      const std::string f = entry.path().filename().string();
      if (f.rfind("thermal_zone", 0) != 0 || !detail::AllDigits(f.substr(12))) continue;
      ThermalZoneSample z;
      z.zone_id = static_cast<int>(detail::ParseStrictUint64(f.substr(12), "zone_id"));
      z.type = detail::ReadSingleLineFile(entry.path() / "type");
      z.temp_millideg_c = detail::ParseStrictInt64(
          detail::ReadSingleLineFile(entry.path() / "temp"), f + "/temp");
      z.temp_celsius = static_cast<double>(z.temp_millideg_c) / 1000.0;
      out->push_back(z);
    }
    std::sort(out->begin(), out->end(),
              [](const ThermalZoneSample& a, const ThermalZoneSample& b) {
                return a.zone_id < b.zone_id;
              });
  }

  void ParseCpuFreq(std::vector<CpuFreqSample>* out) const {
    const auto dir = sysroot_ / "sys" / "devices" / "system" / "cpu";
    if (!std::filesystem::exists(dir)) return;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
      const std::string f = entry.path().filename().string();
      if (f.rfind("cpu", 0) != 0 || f.size() <= 3 || !detail::AllDigits(f.substr(3))) continue;
      const auto fdir = entry.path() / "cpufreq";
      if (!std::filesystem::exists(fdir)) continue;
      CpuFreqSample s;
      s.cpu_id = static_cast<int>(detail::ParseStrictUint64(f.substr(3), "cpu_id"));
      s.scaling_cur_freq_khz = detail::ParseStrictUint64(
          detail::ReadSingleLineFile(fdir / "scaling_cur_freq"), "scaling_cur_freq");
      s.scaling_max_freq_khz = detail::ParseStrictUint64(
          detail::ReadSingleLineFile(fdir / "scaling_max_freq"), "scaling_max_freq");
      if (s.scaling_max_freq_khz == 0) throw std::runtime_error("Zero scaling_max_freq");
      s.utilization_of_max =
          static_cast<double>(s.scaling_cur_freq_khz) / static_cast<double>(s.scaling_max_freq_khz);
      out->push_back(s);
    }
    std::sort(out->begin(), out->end(),
              [](const CpuFreqSample& a, const CpuFreqSample& b) { return a.cpu_id < b.cpu_id; });
  }

  std::filesystem::path sysroot_;
};

}  // namespace sysfs_sampler

#endif  // WAVE_SYSFS_SAMPLER_HPP_
