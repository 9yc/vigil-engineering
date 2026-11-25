// CLI wrapper for proc_watch: run commands under wall-clock and RSS budgets
// or inspect synthetic `/proc` process trees for deterministic testing.
#include "proc_watchdog.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

bool parse_u64_arg(const char* raw, uint64_t& out) {
  if (!raw || *raw == '\0' || *raw == '-') {
    return false;
  }
  errno = 0;
  char* end = nullptr;
  unsigned long long val = std::strtoull(raw, &end, 10);
  if (errno != 0 || end == raw || *end != '\0') {
    return false;
  }
  out = static_cast<uint64_t>(val);
  return true;
}

bool parse_pid_arg(const char* raw, pid_t& out) {
  uint64_t v = 0;
  if (!parse_u64_arg(raw, v) || v == 0 || v > 4194304ULL) {
    return false;
  }
  out = static_cast<pid_t>(v);
  return true;
}

bool write_exclusive_file(const std::string& path, const std::string& content, std::string& err) {
  int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
  if (fd < 0) {
    err = std::string("cannot create output file '") + path + "': " + std::strerror(errno);
    return false;
  }
  const char* ptr = content.data();
  size_t rem = content.size();
  while (rem > 0) {
    ssize_t w = write(fd, ptr, rem);
    if (w > 0) {
      ptr += w;
      rem -= static_cast<size_t>(w);
    } else if (errno != EINTR) {
      err = std::string("write failed: ") + std::strerror(errno);
      close(fd);
      return false;
    }
  }
  if (close(fd) != 0) {
    err = std::string("close failed: ") + std::strerror(errno);
    return false;
  }
  return true;
}

std::string format_report_json(const proc_watch::WatchReport& r) {
  std::ostringstream os;
  const double wall_ms = static_cast<double>(r.wall_time_ns) / 1e6;
  const double tree_cpu_s =
      static_cast<double>(r.final_tree_utime_ticks + r.final_tree_stime_ticks) /
      static_cast<double>(r.clk_tck);
  os.setf(std::ios::fixed);
  os.precision(3);
  os << "{\n"
     << "  \"exit_code\": " << r.exit_code << ",\n"
     << "  \"term_signal\": " << r.term_signal << ",\n"
     << "  \"timed_out\": " << (r.timed_out ? "true" : "false") << ",\n"
     << "  \"rss_exceeded\": " << (r.rss_exceeded ? "true" : "false") << ",\n"
     << "  \"wall_time_ms\": " << wall_ms << ",\n"
     << "  \"peak_rss_kb\": " << r.peak_rss_kb << ",\n"
     << "  \"peak_vsize_kb\": " << r.peak_vsize_kb << ",\n"
     << "  \"peak_processes\": " << r.peak_processes << ",\n"
     << "  \"peak_threads\": " << r.peak_threads << ",\n"
     << "  \"tree_utime_ticks\": " << r.final_tree_utime_ticks << ",\n"
     << "  \"tree_stime_ticks\": " << r.final_tree_stime_ticks << ",\n"
     << "  \"tree_cpu_s\": " << tree_cpu_s << ",\n"
     << "  \"child_ru_utime_us\": " << r.child_ru_utime_us << ",\n"
     << "  \"child_ru_stime_us\": " << r.child_ru_stime_us << ",\n"
     << "  \"child_ru_maxrss_kb\": " << r.child_ru_maxrss_kb << ",\n"
     << "  \"child_ru_minflt\": " << r.child_ru_minflt << ",\n"
     << "  \"child_ru_majflt\": " << r.child_ru_majflt << ",\n"
     << "  \"samples\": " << r.sample_count << "\n"
     << "}\n";
  return os.str();
}

std::string format_trace_csv(const std::vector<proc_watch::TreeSample>& trace) {
  std::ostringstream os;
  os << "elapsed_ns,processes,threads,rss_kb,vsize_kb,utime_ticks,stime_ticks,minflt,majflt\n";
  for (const auto& s : trace) {
    os << s.elapsed_ns << ","
       << s.process_count << ","
       << s.total_threads << ","
       << s.rss_kb << ","
       << s.vsize_kb << ","
       << s.utime_ticks << ","
       << s.stime_ticks << ","
       << s.minflt << ","
       << s.majflt << "\n";
  }
  return os.str();
}

void print_usage(const char* prog) {
  std::fprintf(
      stderr,
      "usage:\n"
      "  %s run [--poll-ms N] [--timeout-ms N] [--max-rss-kb N] [--grace-ms N]\n"
      "         [--report FILE] [--trace-csv FILE] -- COMMAND [ARGS...]\n"
      "  %s snapshot --proc-root DIR --root-pid PID [--page-size BYTES] [--output FILE]\n",
      prog,
      prog);
}

int cmd_snapshot(int argc, char** argv) {
  std::string proc_root = "/proc";
  pid_t root_pid = 0;
  uint64_t page_size = 4096;
  std::string output_path;

  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--proc-root" && i + 1 < argc) {
      proc_root = argv[++i];
    } else if (arg == "--root-pid" && i + 1 < argc) {
      if (!parse_pid_arg(argv[++i], root_pid)) {
        std::fprintf(stderr, "proc_tree_watch: invalid --root-pid\n");
        return 2;
      }
    } else if (arg == "--page-size" && i + 1 < argc) {
      if (!parse_u64_arg(argv[++i], page_size) || page_size < 1024 || (page_size % 1024) != 0) {
        std::fprintf(stderr, "proc_tree_watch: invalid --page-size\n");
        return 2;
      }
    } else if (arg == "--output" && i + 1 < argc) {
      output_path = argv[++i];
    } else {
      std::fprintf(stderr, "proc_tree_watch: unknown snapshot option '%s'\n", arg.c_str());
      return 2;
    }
  }
  if (root_pid <= 0) {
    std::fprintf(stderr, "proc_tree_watch: --root-pid is required\n");
    return 2;
  }

  proc_watch::TreeSample s = proc_watch::sample_process_tree(
      proc_root, root_pid, 0, 0, static_cast<long>(page_size));
  if (s.process_count == 0) {
    std::fprintf(stderr, "proc_tree_watch: root pid %d not found in %s\n",
                 static_cast<int>(root_pid), proc_root.c_str());
    return 1;
  }

  std::ostringstream os;
  os << "{\n"
     << "  \"processes\": " << s.process_count << ",\n"
     << "  \"threads\": " << s.total_threads << ",\n"
     << "  \"rss_kb\": " << s.rss_kb << ",\n"
     << "  \"vsize_kb\": " << s.vsize_kb << ",\n"
     << "  \"utime_ticks\": " << s.utime_ticks << ",\n"
     << "  \"stime_ticks\": " << s.stime_ticks << ",\n"
     << "  \"minflt\": " << s.minflt << ",\n"
     << "  \"majflt\": " << s.majflt << "\n"
     << "}\n";

  if (!output_path.empty()) {
    std::string err;
    if (!write_exclusive_file(output_path, os.str(), err)) {
      std::fprintf(stderr, "proc_tree_watch: %s\n", err.c_str());
      return 2;
    }
  } else {
    std::fwrite(os.str().data(), 1, os.str().size(), stdout);
  }
  return 0;
}

int cmd_run(int argc, char** argv) {
  proc_watch::WatchConfig cfg;
  std::string report_path;
  std::string trace_path;
  int sep_idx = -1;

  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--") {
      sep_idx = i;
      break;
    } else if (arg == "--poll-ms" && i + 1 < argc) {
      if (!parse_u64_arg(argv[++i], cfg.poll_interval_ms) || cfg.poll_interval_ms == 0) {
        std::fprintf(stderr, "proc_tree_watch: invalid --poll-ms\n");
        return 2;
      }
    } else if (arg == "--timeout-ms" && i + 1 < argc) {
      if (!parse_u64_arg(argv[++i], cfg.timeout_ms)) {
        std::fprintf(stderr, "proc_tree_watch: invalid --timeout-ms\n");
        return 2;
      }
    } else if (arg == "--max-rss-kb" && i + 1 < argc) {
      if (!parse_u64_arg(argv[++i], cfg.max_rss_kb)) {
        std::fprintf(stderr, "proc_tree_watch: invalid --max-rss-kb\n");
        return 2;
      }
    } else if (arg == "--grace-ms" && i + 1 < argc) {
      if (!parse_u64_arg(argv[++i], cfg.kill_grace_ms)) {
        std::fprintf(stderr, "proc_tree_watch: invalid --grace-ms\n");
        return 2;
      }
    } else if (arg == "--report" && i + 1 < argc) {
      report_path = argv[++i];
    } else if (arg == "--trace-csv" && i + 1 < argc) {
      trace_path = argv[++i];
    } else {
      std::fprintf(stderr, "proc_tree_watch: unknown run option '%s'\n", arg.c_str());
      return 2;
    }
  }

  if (sep_idx < 0 || sep_idx + 1 >= argc) {
    std::fprintf(stderr, "proc_tree_watch: expected '-- COMMAND [ARGS...]'\n");
    return 2;
  }
  // Check output destinations do not already exist before executing child command.
  if (!report_path.empty() && access(report_path.c_str(), F_OK) == 0) {
    std::fprintf(stderr, "proc_tree_watch: report file already exists: %s\n", report_path.c_str());
    return 2;
  }
  if (!trace_path.empty() && access(trace_path.c_str(), F_OK) == 0) {
    std::fprintf(stderr, "proc_tree_watch: trace file already exists: %s\n", trace_path.c_str());
    return 2;
  }

  std::vector<std::string> cmd_argv;
  for (int i = sep_idx + 1; i < argc; ++i) {
    cmd_argv.emplace_back(argv[i]);
  }

  proc_watch::WatchReport report;
  std::string err;
  if (!proc_watch::run_watched_command(cmd_argv, cfg, report, err)) {
    std::fprintf(stderr, "proc_tree_watch: %s\n", err.c_str());
    return 2;
  }

  std::string json_str = format_report_json(report);
  if (!trace_path.empty()) {
    if (!write_exclusive_file(trace_path, format_trace_csv(report.trace), err)) {
      std::fprintf(stderr, "proc_tree_watch: %s\n", err.c_str());
      return 2;
    }
  }
  if (!report_path.empty()) {
    if (!write_exclusive_file(report_path, json_str, err)) {
      std::fprintf(stderr, "proc_tree_watch: %s\n", err.c_str());
      return 2;
    }
  } else {
    std::fwrite(json_str.data(), 1, json_str.size(), stdout);
  }

  if (report.timed_out) {
    return 124;
  }
  if (report.rss_exceeded) {
    return 125;
  }
  return report.exit_code;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    print_usage(argv[0]);
    return 2;
  }
  std::string sub = argv[1];
  if (sub == "run") {
    return cmd_run(argc, argv);
  }
  if (sub == "snapshot") {
    return cmd_snapshot(argc, argv);
  }
  print_usage(argv[0]);
  return 2;
}
