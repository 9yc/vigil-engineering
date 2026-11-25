// Process-tree resource sampler and budget watchdog for Linux `/proc`.
// Walks `/proc/<pid>/stat` across an entire descendant process tree or process
// group, aggregates RSS, CPU ticks, page faults and thread counts, and supports
// bounded wall-clock and RSS enforcement with staged SIGTERM -> SIGKILL cleanup.
#ifndef PROC_WATCHDOG_HPP
#define PROC_WATCHDOG_HPP

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <map>
#include <set>
#include <string>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <vector>

namespace proc_watch {

struct ProcNodeStat {
  pid_t pid = 0;
  pid_t ppid = 0;
  pid_t pgrp = 0;
  char state = '?';
  uint64_t minflt = 0;
  uint64_t majflt = 0;
  uint64_t utime_ticks = 0;
  uint64_t stime_ticks = 0;
  long num_threads = 0;
  int64_t rss_pages = 0;
  uint64_t vsize_bytes = 0;
};

struct TreeSample {
  uint64_t elapsed_ns = 0;
  size_t process_count = 0;
  size_t total_threads = 0;
  uint64_t rss_kb = 0;
  uint64_t vsize_kb = 0;
  uint64_t utime_ticks = 0;
  uint64_t stime_ticks = 0;
  uint64_t minflt = 0;
  uint64_t majflt = 0;
};

struct WatchConfig {
  uint64_t poll_interval_ms = 20;
  uint64_t timeout_ms = 0;       // 0 = unlimited
  uint64_t max_rss_kb = 0;       // 0 = unlimited
  uint64_t kill_grace_ms = 200;
  std::string proc_root = "/proc";
};

struct WatchReport {
  int exit_code = -1;
  int term_signal = 0;
  bool timed_out = false;
  bool rss_exceeded = false;
  uint64_t wall_time_ns = 0;
  uint64_t peak_rss_kb = 0;
  uint64_t peak_vsize_kb = 0;
  size_t peak_processes = 0;
  size_t peak_threads = 0;
  uint64_t final_tree_utime_ticks = 0;
  uint64_t final_tree_stime_ticks = 0;
  uint64_t child_ru_utime_us = 0;
  uint64_t child_ru_stime_us = 0;
  uint64_t child_ru_maxrss_kb = 0;
  uint64_t child_ru_minflt = 0;
  uint64_t child_ru_majflt = 0;
  size_t sample_count = 0;
  long clk_tck = 100;
  std::vector<TreeSample> trace;
};

inline uint64_t monotonic_ns() {
  struct timespec ts {};
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
    return 0;
  }
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL +
         static_cast<uint64_t>(ts.tv_nsec);
}

inline void sleep_ms(uint64_t ms) {
  struct timespec req {};
  req.tv_sec = static_cast<time_t>(ms / 1000ULL);
  req.tv_nsec = static_cast<long>((ms % 1000ULL) * 1000000ULL);
  while (nanosleep(&req, &req) != 0 && errno == EINTR) {
  }
}

inline bool read_small_file(const std::string& path, std::string& out) {
  int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    return false;
  }
  char buf[4096];
  out.clear();
  while (true) {
    ssize_t n = read(fd, buf, sizeof(buf));
    if (n > 0) {
      out.append(buf, static_cast<size_t>(n));
      if (out.size() > 65536) {
        close(fd);
        return false;
      }
    } else if (n == 0) {
      break;
    } else if (errno != EINTR) {
      close(fd);
      return false;
    }
  }
  close(fd);
  return !out.empty();
}

// Parse Linux `/proc/<pid>/stat` safely even when `(comm)` contains spaces or ')'.
inline bool parse_proc_stat_line(const std::string& content, ProcNodeStat& st) {
  const size_t open_paren = content.find('(');
  const size_t close_paren = content.rfind(')');
  if (open_paren == std::string::npos || close_paren == std::string::npos ||
      close_paren <= open_paren || close_paren + 2 >= content.size()) {
    return false;
  }
  char* end_pid = nullptr;
  errno = 0;
  long pid_val = std::strtol(content.c_str(), &end_pid, 10);
  if (errno != 0 || end_pid == content.c_str() || pid_val <= 0) {
    return false;
  }
  st.pid = static_cast<pid_t>(pid_val);

  // Tokenize fields after ") ":
  // [0]=state, [1]=ppid, [2]=pgrp, [7]=minflt, [9]=majflt,
  // [11]=utime, [12]=stime, [17]=num_threads, [20]=vsize, [21]=rss
  std::vector<std::string> fields;
  fields.reserve(32);
  const char* p = content.c_str() + close_paren + 1;
  while (*p != '\0') {
    while (*p != '\0' && std::isspace(static_cast<unsigned char>(*p))) {
      ++p;
    }
    if (*p == '\0') {
      break;
    }
    const char* start = p;
    while (*p != '\0' && !std::isspace(static_cast<unsigned char>(*p))) {
      ++p;
    }
    fields.emplace_back(start, static_cast<size_t>(p - start));
  }
  if (fields.size() < 22 || fields[0].empty()) {
    return false;
  }

  auto parse_i64 = [](const std::string& tok, int64_t& dst) -> bool {
    errno = 0;
    char* end = nullptr;
    long long v = std::strtoll(tok.c_str(), &end, 10);
    if (errno != 0 || end == tok.c_str() || *end != '\0') {
      return false;
    }
    dst = static_cast<int64_t>(v);
    return true;
  };
  auto parse_u64 = [](const std::string& tok, uint64_t& dst) -> bool {
    if (!tok.empty() && tok[0] == '-') {
      return false;
    }
    errno = 0;
    char* end = nullptr;
    unsigned long long v = std::strtoull(tok.c_str(), &end, 10);
    if (errno != 0 || end == tok.c_str() || *end != '\0') {
      return false;
    }
    dst = static_cast<uint64_t>(v);
    return true;
  };

  int64_t ppid = 0, pgrp = 0, threads = 0, rss = 0;
  if (!parse_i64(fields[1], ppid) || !parse_i64(fields[2], pgrp) ||
      !parse_u64(fields[7], st.minflt) || !parse_u64(fields[9], st.majflt) ||
      !parse_u64(fields[11], st.utime_ticks) || !parse_u64(fields[12], st.stime_ticks) ||
      !parse_i64(fields[17], threads) || !parse_u64(fields[20], st.vsize_bytes) ||
      !parse_i64(fields[21], rss)) {
    return false;
  }
  st.state = fields[0][0];
  st.ppid = static_cast<pid_t>(ppid);
  st.pgrp = static_cast<pid_t>(pgrp);
  st.num_threads = static_cast<long>(std::max<int64_t>(0, threads));
  st.rss_pages = std::max<int64_t>(0, rss);
  return true;
}

inline bool is_numeric_name(const char* s) {
  if (*s == '\0') {
    return false;
  }
  for (const char* p = s; *p != '\0'; ++p) {
    if (!std::isdigit(static_cast<unsigned char>(*p))) {
      return false;
    }
  }
  return true;
}

inline TreeSample sample_process_tree(
    const std::string& proc_root,
    pid_t root_pid,
    pid_t target_pgrp,
    uint64_t elapsed_ns,
    long page_size_bytes) {
  TreeSample agg {};
  agg.elapsed_ns = elapsed_ns;

  DIR* dir = opendir(proc_root.c_str());
  if (!dir) {
    return agg;
  }

  std::map<pid_t, ProcNodeStat> all_nodes;
  std::map<pid_t, std::vector<pid_t>> children_map;
  struct dirent* ent = nullptr;
  std::string buf;
  while ((ent = readdir(dir)) != nullptr) {
    if (!is_numeric_name(ent->d_name)) {
      continue;
    }
    std::string stat_path = proc_root + "/" + ent->d_name + "/stat";
    if (!read_small_file(stat_path, buf)) {
      continue;
    }
    ProcNodeStat node {};
    if (!parse_proc_stat_line(buf, node)) {
      continue;
    }
    if (node.state == 'Z') {
      continue;
    }
    all_nodes[node.pid] = node;
    children_map[node.ppid].push_back(node.pid);
  }
  closedir(dir);

  std::set<pid_t> visited;
  std::vector<pid_t> stack;
  if (all_nodes.count(root_pid)) {
    stack.push_back(root_pid);
    visited.insert(root_pid);
  }
  if (target_pgrp > 0) {
    for (const auto& kv : all_nodes) {
      if (kv.second.pgrp == target_pgrp && !visited.count(kv.first)) {
        visited.insert(kv.first);
        stack.push_back(kv.first);
      }
    }
  }
  while (!stack.empty()) {
    pid_t cur = stack.back();
    stack.pop_back();
    auto it = children_map.find(cur);
    if (it != children_map.end()) {
      for (pid_t child_pid : it->second) {
        if (!visited.count(child_pid)) {
          visited.insert(child_pid);
          stack.push_back(child_pid);
        }
      }
    }
  }

  const uint64_t page_kb = static_cast<uint64_t>(std::max<long>(1024, page_size_bytes)) / 1024ULL;
  for (pid_t pid : visited) {
    const ProcNodeStat& n = all_nodes[pid];
    agg.process_count += 1;
    agg.total_threads += static_cast<size_t>(n.num_threads);
    agg.rss_kb += static_cast<uint64_t>(n.rss_pages) * page_kb;
    agg.vsize_kb += n.vsize_bytes / 1024ULL;
    agg.utime_ticks += n.utime_ticks;
    agg.stime_ticks += n.stime_ticks;
    agg.minflt += n.minflt;
    agg.majflt += n.majflt;
  }
  return agg;
}

inline void terminate_process_group(pid_t pgid, uint64_t grace_ms, int& status_out, struct rusage& ru_out) {
  if (pgid <= 0) {
    return;
  }
  kill(-pgid, SIGTERM);
  const uint64_t deadline = monotonic_ns() + grace_ms * 1000000ULL;
  while (true) {
    pid_t rc = wait4(pgid, &status_out, WNOHANG, &ru_out);
    if (rc == pgid) {
      kill(-pgid, SIGKILL);
      return;
    }
    if (rc < 0 && errno != EINTR) {
      return;
    }
    if (monotonic_ns() >= deadline) {
      break;
    }
    sleep_ms(10);
  }
  kill(-pgid, SIGKILL);
  while (wait4(pgid, &status_out, 0, &ru_out) < 0 && errno == EINTR) {
  }
}

inline bool run_watched_command(
    const std::vector<std::string>& argv,
    const WatchConfig& cfg,
    WatchReport& report,
    std::string& err) {
  if (argv.empty()) {
    err = "command argv must not be empty";
    return false;
  }
  if (cfg.poll_interval_ms == 0) {
    err = "poll_interval_ms must be positive";
    return false;
  }

  int exec_err_pipe[2] = {-1, -1};
  if (pipe(exec_err_pipe) != 0) {
    err = std::string("pipe failed: ") + std::strerror(errno);
    return false;
  }
  fcntl(exec_err_pipe[0], F_SETFD, FD_CLOEXEC);
  fcntl(exec_err_pipe[1], F_SETFD, FD_CLOEXEC);

  const uint64_t start_ns = monotonic_ns();
  pid_t child = fork();
  if (child < 0) {
    err = std::string("fork failed: ") + std::strerror(errno);
    close(exec_err_pipe[0]);
    close(exec_err_pipe[1]);
    return false;
  }

  if (child == 0) {
    close(exec_err_pipe[0]);
    (void)setpgid(0, 0);
    std::vector<char*> c_argv;
    c_argv.reserve(argv.size() + 1);
    for (const auto& s : argv) {
      c_argv.push_back(const_cast<char*>(s.c_str()));
    }
    c_argv.push_back(nullptr);
    execvp(c_argv[0], c_argv.data());
    int code = errno;
    ssize_t ignored = write(exec_err_pipe[1], &code, sizeof(code));
    if (ignored < 0) {
    }
    _exit(127);
  }

  close(exec_err_pipe[1]);
  (void)setpgid(child, child);

  int child_exec_errno = 0;
  ssize_t nread = read(exec_err_pipe[0], &child_exec_errno, sizeof(child_exec_errno));
  close(exec_err_pipe[0]);
  if (nread == sizeof(child_exec_errno) && child_exec_errno != 0) {
    int status = 0;
    while (waitpid(child, &status, 0) < 0 && errno == EINTR) {
    }
    err = std::string("execvp failed: ") + std::strerror(child_exec_errno);
    return false;
  }

  long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0) {
    page_size = 4096;
  }
  long clk_tck = sysconf(_SC_CLK_TCK);
  if (clk_tck <= 0) {
    clk_tck = 100;
  }
  report = WatchReport{};
  report.clk_tck = clk_tck;

  int status = 0;
  struct rusage ru {};
  bool reaped = false;

  while (true) {
    uint64_t now_ns = monotonic_ns();
    uint64_t elapsed_ns = (now_ns >= start_ns) ? (now_ns - start_ns) : 0;

    pid_t w = wait4(child, &status, WNOHANG, &ru);
    if (w == child) {
      reaped = true;
      break;
    }
    if (w < 0 && errno != EINTR) {
      err = std::string("wait4 failed: ") + std::strerror(errno);
      return false;
    }

    TreeSample sample = sample_process_tree(cfg.proc_root, child, child, elapsed_ns, page_size);
    if (sample.process_count > 0) {
      report.sample_count += 1;
      report.peak_rss_kb = std::max(report.peak_rss_kb, sample.rss_kb);
      report.peak_vsize_kb = std::max(report.peak_vsize_kb, sample.vsize_kb);
      report.peak_processes = std::max(report.peak_processes, sample.process_count);
      report.peak_threads = std::max(report.peak_threads, sample.total_threads);
      report.final_tree_utime_ticks = std::max(report.final_tree_utime_ticks, sample.utime_ticks);
      report.final_tree_stime_ticks = std::max(report.final_tree_stime_ticks, sample.stime_ticks);
      report.trace.push_back(sample);
    }

    if (cfg.max_rss_kb > 0 && sample.rss_kb > cfg.max_rss_kb) {
      report.rss_exceeded = true;
      terminate_process_group(child, cfg.kill_grace_ms, status, ru);
      reaped = true;
      break;
    }
    if (cfg.timeout_ms > 0 && elapsed_ns >= cfg.timeout_ms * 1000000ULL) {
      report.timed_out = true;
      terminate_process_group(child, cfg.kill_grace_ms, status, ru);
      reaped = true;
      break;
    }

    sleep_ms(cfg.poll_interval_ms);
  }

  if (!reaped) {
    while (wait4(child, &status, 0, &ru) < 0 && errno == EINTR) {
    }
  }
  // Ensure no orphan grandchildren survive in the child process group.
  kill(-child, SIGKILL);

  uint64_t end_ns = monotonic_ns();
  report.wall_time_ns = (end_ns >= start_ns) ? (end_ns - start_ns) : 0;
  if (WIFEXITED(status)) {
    report.exit_code = WEXITSTATUS(status);
    report.term_signal = 0;
  } else if (WIFSIGNALED(status)) {
    report.term_signal = WTERMSIG(status);
    report.exit_code = 128 + report.term_signal;
  }

  report.child_ru_utime_us = static_cast<uint64_t>(ru.ru_utime.tv_sec) * 1000000ULL +
                             static_cast<uint64_t>(ru.ru_utime.tv_usec);
  report.child_ru_stime_us = static_cast<uint64_t>(ru.ru_stime.tv_sec) * 1000000ULL +
                             static_cast<uint64_t>(ru.ru_stime.tv_usec);
  // On Linux ru_maxrss is in kilobytes.
  report.child_ru_maxrss_kb = (ru.ru_maxrss > 0) ? static_cast<uint64_t>(ru.ru_maxrss) : 0ULL;
  report.peak_rss_kb = std::max(report.peak_rss_kb, report.child_ru_maxrss_kb);
  report.child_ru_minflt = (ru.ru_minflt > 0) ? static_cast<uint64_t>(ru.ru_minflt) : 0ULL;
  report.child_ru_majflt = (ru.ru_majflt > 0) ? static_cast<uint64_t>(ru.ru_majflt) : 0ULL;
  return true;
}

}  // namespace proc_watch

#endif  // PROC_WATCHDOG_HPP
