#include "sysfs_sampler.hpp"

#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

std::string EscapeJson(const std::string& s) {
  std::ostringstream oss;
  for (char c : s) {
    switch (c) {
      case '"': oss << "\\\""; break;
      case '\\': oss << "\\\\"; break;
      case '\n': oss << "\\n"; break;
      case '\r': oss << "\\r"; break;
      case '\t': oss << "\\t"; break;
      default: oss << c; break;
    }
  }
  return oss.str();
}

void WriteExclusiveFile(const std::string& path, const std::string& content) {
  int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
  if (fd < 0) {
    throw std::runtime_error("Cannot create output file '" + path +
                             "': " + std::strerror(errno));
  }
  size_t off = 0;
  while (off < content.size()) {
    ssize_t n = ::write(fd, content.data() + off, content.size() - off);
    if (n < 0) {
      int err = errno;
      ::close(fd);
      throw std::runtime_error("Write failed: " + std::string(std::strerror(err)));
    }
    off += static_cast<size_t>(n);
  }
  ::close(fd);
}

void RenderCpuJiffies(std::ostringstream& out, const sysfs_sampler::CpuJiffies& c) {
  out << "{\"label\":\"" << EscapeJson(c.label) << "\""
      << ",\"cpu_index\":" << c.cpu_index
      << ",\"user\":" << c.user
      << ",\"nice\":" << c.nice
      << ",\"system\":" << c.system
      << ",\"idle\":" << c.idle
      << ",\"iowait\":" << c.iowait
      << ",\"irq\":" << c.irq
      << ",\"softirq\":" << c.softirq
      << ",\"steal\":" << c.steal
      << ",\"busy_jiffies\":" << c.BusyAll()
      << ",\"total_jiffies\":" << c.Total() << "}";
}

void RenderPsiWindow(std::ostringstream& out, const sysfs_sampler::PsiWindow& w) {
  out << "{\"present\":" << (w.present ? "true" : "false")
      << ",\"avg10\":" << w.avg10
      << ",\"avg60\":" << w.avg60
      << ",\"avg300\":" << w.avg300
      << ",\"total_us\":" << w.total_us << "}";
}

void RenderSnapshot(std::ostringstream& out,
                    const sysfs_sampler::TelemetrySnapshot& snap) {
  out << "{\n    \"meminfo_kb\": {";
  bool first = true;
  for (const auto& [k, v] : snap.meminfo_kb) {
    if (!first) out << ", ";
    first = false;
    out << "\"" << EscapeJson(k) << "\": " << v;
  }
  out << "},\n    \"cpu_total\": ";
  RenderCpuJiffies(out, snap.total_cpu);
  out << ",\n    \"per_cpu\": [";
  for (size_t i = 0; i < snap.per_cpu.size(); ++i) {
    if (i > 0) out << ", ";
    RenderCpuJiffies(out, snap.per_cpu[i]);
  }
  out << "],\n    \"process\": ";
  if (snap.process.has_value()) {
    const auto& p = *snap.process;
    out << "{\"pid\":" << p.pid
        << ",\"name\":\"" << EscapeJson(p.name) << "\""
        << ",\"state\":\"" << EscapeJson(p.state) << "\""
        << ",\"vm_rss_kb\":" << p.vm_rss_kb
        << ",\"vm_peak_kb\":" << p.vm_peak_kb
        << ",\"threads\":" << p.threads
        << ",\"voluntary_ctxt_switches\":" << p.voluntary_ctxt_switches
        << ",\"nonvoluntary_ctxt_switches\":" << p.nonvoluntary_ctxt_switches << "}";
  } else {
    out << "null";
  }
  out << ",\n    \"pressure\": {";
  first = true;
  for (const auto& [res, psi] : snap.pressure) {
    if (!first) out << ", ";
    first = false;
    out << "\"" << EscapeJson(res) << "\": {\"some\": ";
    RenderPsiWindow(out, psi.some);
    out << ", \"full\": ";
    RenderPsiWindow(out, psi.full);
    out << "}";
  }
  out << "},\n    \"thermal_zones\": [";
  for (size_t i = 0; i < snap.thermal_zones.size(); ++i) {
    if (i > 0) out << ", ";
    const auto& tz = snap.thermal_zones[i];
    out << "{\"zone_id\":" << tz.zone_id
        << ",\"type\":\"" << EscapeJson(tz.type) << "\""
        << ",\"temp_millideg_c\":" << tz.temp_millideg_c
        << ",\"temp_celsius\":" << tz.temp_celsius << "}";
  }
  out << "],\n    \"cpufreq\": [";
  for (size_t i = 0; i < snap.cpufreq.size(); ++i) {
    if (i > 0) out << ", ";
    const auto& cf = snap.cpufreq[i];
    out << "{\"cpu_id\":" << cf.cpu_id
        << ",\"scaling_cur_freq_khz\":" << cf.scaling_cur_freq_khz
        << ",\"scaling_max_freq_khz\":" << cf.scaling_max_freq_khz
        << ",\"utilization_of_max\":" << cf.utilization_of_max << "}";
  }
  out << "]\n  }";
}

void RenderCpuDelta(std::ostringstream& out,
                    const sysfs_sampler::CpuUtilizationDelta& d) {
  out << "{\"label\":\"" << EscapeJson(d.label) << "\""
      << ",\"cpu_index\":" << d.cpu_index
      << ",\"delta_total\":" << d.delta_total
      << ",\"delta_busy\":" << d.delta_busy
      << ",\"delta_idle\":" << d.delta_idle
      << ",\"delta_iowait\":" << d.delta_iowait
      << ",\"busy_fraction\":" << d.busy_fraction
      << ",\"user_fraction\":" << d.user_fraction
      << ",\"system_fraction\":" << d.system_fraction
      << ",\"iowait_fraction\":" << d.iowait_fraction
      << ",\"irq_fraction\":" << d.irq_fraction << "}";
}

void RenderDelta(std::ostringstream& out, const sysfs_sampler::TelemetryDelta& d) {
  out << "{\n    \"cpu_total\": ";
  RenderCpuDelta(out, d.total_cpu);
  out << ",\n    \"per_cpu\": [";
  for (size_t i = 0; i < d.per_cpu.size(); ++i) {
    if (i > 0) out << ", ";
    RenderCpuDelta(out, d.per_cpu[i]);
  }
  out << "],\n    \"delta_voluntary_ctxt_switches\": ";
  if (d.delta_voluntary_ctxt_switches.has_value()) out << *d.delta_voluntary_ctxt_switches;
  else out << "null";
  out << ",\n    \"delta_nonvoluntary_ctxt_switches\": ";
  if (d.delta_nonvoluntary_ctxt_switches.has_value()) out << *d.delta_nonvoluntary_ctxt_switches;
  else out << "null";
  out << ",\n    \"psi_some_delta_us\": {";
  bool first = true;
  for (const auto& [k, v] : d.psi_some_delta_us) {
    if (!first) out << ", ";
    first = false;
    out << "\"" << EscapeJson(k) << "\": " << v;
  }
  out << "},\n    \"psi_full_delta_us\": {";
  first = true;
  for (const auto& [k, v] : d.psi_full_delta_us) {
    if (!first) out << ", ";
    first = false;
    out << "\"" << EscapeJson(k) << "\": " << v;
  }
  out << "}\n  }";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string sysroot = "/", compare_sysroot, output_path;
    std::optional<int32_t> pid;
    uint64_t interval_ms = 50, samples = 1;

    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      auto next_val = [&](const std::string& flag) -> std::string {
        if (i + 1 >= argc) throw std::runtime_error("Missing value for " + flag);
        return argv[++i];
      };
      if (arg == "--sysroot") {
        sysroot = next_val(arg);
      } else if (arg == "--compare-sysroot") {
        compare_sysroot = next_val(arg);
      } else if (arg == "--pid") {
        int64_t p = sysfs_sampler::detail::ParseStrictInt64(next_val(arg), "--pid");
        if (p <= 0 || p > std::numeric_limits<int32_t>::max()) {
          throw std::runtime_error("--pid out of valid int32 range");
        }
        pid = static_cast<int32_t>(p);
      } else if (arg == "--interval-ms") {
        interval_ms = sysfs_sampler::detail::ParseStrictUint64(next_val(arg), "--interval-ms");
      } else if (arg == "--samples") {
        samples = sysfs_sampler::detail::ParseStrictUint64(next_val(arg), "--samples");
        if (samples == 0) throw std::runtime_error("--samples must be >= 1");
      } else if (arg == "--output") {
        output_path = next_val(arg);
      } else {
        throw std::runtime_error("Unknown CLI argument: " + arg);
      }
    }

    std::vector<sysfs_sampler::TelemetrySnapshot> snapshots;
    if (!compare_sysroot.empty()) {
      snapshots.push_back(sysfs_sampler::SysfsSampler(sysroot).Sample(pid));
      snapshots.push_back(sysfs_sampler::SysfsSampler(compare_sysroot).Sample(pid));
    } else {
      sysfs_sampler::SysfsSampler sampler(sysroot);
      snapshots.reserve(samples);
      for (uint64_t s = 0; s < samples; ++s) {
        if (s > 0 && interval_ms > 0) {
          std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
        }
        snapshots.push_back(sampler.Sample(pid));
      }
    }

    std::vector<sysfs_sampler::TelemetryDelta> deltas;
    for (size_t i = 1; i < snapshots.size(); ++i) {
      deltas.push_back(sysfs_sampler::SysfsSampler::ComputeDelta(snapshots[i - 1], snapshots[i]));
    }

    std::ostringstream out;
    out << std::fixed << std::setprecision(6)
        << "{\n  \"sample_count\": " << snapshots.size() << ",\n  \"snapshots\": [";
    for (size_t i = 0; i < snapshots.size(); ++i) {
      if (i > 0) out << ", ";
      RenderSnapshot(out, snapshots[i]);
    }
    out << "],\n  \"deltas\": [";
    for (size_t i = 0; i < deltas.size(); ++i) {
      if (i > 0) out << ", ";
      RenderDelta(out, deltas[i]);
    }
    out << "]\n}\n";

    if (output_path.empty()) std::cout << out.str();
    else WriteExclusiveFile(output_path, out.str());
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "proc_telemetry_dump error: " << ex.what() << "\n";
    return 1;
  }
}
