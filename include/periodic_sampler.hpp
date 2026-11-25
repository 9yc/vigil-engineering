#ifndef PERIODIC_SAMPLER_HPP_
#define PERIODIC_SAMPLER_HPP_

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <poll.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace periodic_sampler {

constexpr int64_t kNsPerMs = 1000000LL;
constexpr int64_t kNsPerSec = 1000000000LL;

struct TimerSlackInfo {
    int64_t previous_ns = -1;
    int64_t requested_ns = 0;
    int64_t effective_ns = -1;
    int error_number = 0;
};

struct ProbeSpec {
    std::vector<std::string> argv;
    std::string shell_cmd;
    int64_t timeout_ns = 5000LL * kNsPerMs;
    size_t max_stdout_bytes = 4U * 1024U * 1024U;
    bool merge_stderr = false;
};

struct SampleOutcome {
    uint64_t sample_index = 0;
    int64_t scheduled_ns = 0;
    int64_t dispatch_ns = 0;
    int64_t end_ns = 0;
    int64_t boottime_ns = 0;
    int64_t realtime_ns = 0;
    int rc = 0;
    bool timed_out = false;
    bool is_final_bracket = false;
    std::string stdout_payload;
};

enum class OverrunPolicy { kCatchUp, kSkipMissed };
enum class StopReason { kNone, kMaxSamples, kDurationLimit, kStopFile, kSignal, kFatalError };

struct SamplerConfig {
    ProbeSpec probe;
    int64_t interval_ns = 100LL * kNsPerMs;
    uint64_t max_samples = 0;
    int64_t max_duration_ns = 0;
    uint64_t timerslack_ns = 0;
    bool set_timerslack = false;
    bool capture_final_on_stop = true;
    bool fail_on_probe_error = false;
    OverrunPolicy overrun_policy = OverrunPolicy::kCatchUp;
    std::string output_path;
    std::string csv_path;
    std::string meta_path;
    std::string stop_file_path;
    std::string done_file_path;
};

struct JitterSummary {
    uint64_t count = 0;
    double mean_lag_ns = 0.0;
    int64_t min_lag_ns = 0;
    int64_t max_lag_ns = 0;
    int64_t p50_lag_ns = 0;
    int64_t p95_lag_ns = 0;
    int64_t p99_lag_ns = 0;
};

struct RunSummary {
    TimerSlackInfo timerslack;
    uint64_t total_samples = 0;
    uint64_t periodic_samples = 0;
    uint64_t nonzero_rc_samples = 0;
    uint64_t timed_out_samples = 0;
    uint64_t skipped_ticks = 0;
    bool final_bracket_captured = false;
    StopReason stop_reason = StopReason::kNone;
    int terminating_signal = 0;
    int64_t run_start_ns = 0;
    int64_t run_end_ns = 0;
    JitterSummary jitter;
    bool succeeded = false;
    std::string error_detail;
};

inline struct timespec ns_to_timespec(int64_t ns) noexcept {
    if (ns < 0) ns = 0;
    struct timespec ts {};
    ts.tv_sec = static_cast<time_t>(ns / kNsPerSec);
    ts.tv_nsec = static_cast<long>(ns % kNsPerSec);
    return ts;
}

inline int64_t read_clock_ns(clockid_t clock_id) {
    struct timespec ts {};
    if (::clock_gettime(clock_id, &ts) != 0) {
        throw std::runtime_error("clock_gettime failed: " + std::string(std::strerror(errno)));
    }
    return static_cast<int64_t>(ts.tv_sec) * kNsPerSec + static_cast<int64_t>(ts.tv_nsec);
}

inline void write_all_fd(int fd, const char* data, size_t len) {
    size_t offset = 0;
    while (offset < len) {
        const ssize_t n = ::write(fd, data + offset, len - offset);
        if (n < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error("write failed: " + std::string(std::strerror(errno)));
        }
        offset += static_cast<size_t>(n);
    }
}

class ExclusiveFileGuard {
public:
    ExclusiveFileGuard() = default;
    explicit ExclusiveFileGuard(const std::string& path, mode_t mode = 0644) { open_exclusive(path, mode); }
    ExclusiveFileGuard(const ExclusiveFileGuard&) = delete;
    ExclusiveFileGuard& operator=(const ExclusiveFileGuard&) = delete;
    ExclusiveFileGuard(ExclusiveFileGuard&& other) noexcept : fd_(other.fd_), path_(std::move(other.path_)) {
        other.fd_ = -1;
    }
    ExclusiveFileGuard& operator=(ExclusiveFileGuard&& other) noexcept {
        if (this != &other) {
            close_quietly();
            fd_ = other.fd_;
            path_ = std::move(other.path_);
            other.fd_ = -1;
        }
        return *this;
    }
    ~ExclusiveFileGuard() { close_quietly(); }

    void open_exclusive(const std::string& path, mode_t mode = 0644) {
        close_quietly();
        const int opened = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, mode);
        if (opened < 0) {
            throw std::runtime_error("exclusive open rejected for '" + path + "': " +
                                     std::string(std::strerror(errno)));
        }
        fd_ = opened;
        path_ = path;
    }

    void write_str(const std::string& content) {
        if (fd_ < 0) throw std::logic_error("ExclusiveFileGuard is not open");
        if (!content.empty()) write_all_fd(fd_, content.data(), content.size());
    }

    void sync_and_close() {
        if (fd_ >= 0) {
            (void)::fsync(fd_);
            const int closed = ::close(fd_);
            fd_ = -1;
            if (closed < 0) {
                throw std::runtime_error("close failed for '" + path_ + "': " +
                                         std::string(std::strerror(errno)));
            }
        }
    }

    bool is_open() const noexcept { return fd_ >= 0; }

private:
    void close_quietly() noexcept {
        if (fd_ >= 0) {
            (void)::close(fd_);
            fd_ = -1;
        }
    }

    int fd_ = -1;
    std::string path_;
};

inline TimerSlackInfo apply_timerslack(bool should_modify, uint64_t requested_ns) noexcept {
    TimerSlackInfo info{};
    info.requested_ns = static_cast<int64_t>(requested_ns);
    errno = 0;
    const long prev = ::prctl(PR_GET_TIMERSLACK, 0UL, 0UL, 0UL, 0UL);
    if (prev < 0) {
        info.error_number = errno;
        return info;
    }
    info.previous_ns = static_cast<int64_t>(prev);
    info.effective_ns = info.previous_ns;
    if (should_modify) {
        const unsigned long clamped = requested_ns == 0 ? 1UL : static_cast<unsigned long>(requested_ns);
        if (::prctl(PR_SET_TIMERSLACK, clamped, 0UL, 0UL, 0UL) != 0) {
            info.error_number = errno;
            return info;
        }
        const long updated = ::prctl(PR_GET_TIMERSLACK, 0UL, 0UL, 0UL, 0UL);
        if (updated < 0) info.error_number = errno;
        else info.effective_ns = static_cast<int64_t>(updated);
    }
    return info;
}

inline bool file_exists_fast(const std::string& path) noexcept {
    return !path.empty() && (::access(path.c_str(), F_OK) == 0);
}

inline bool sleep_until_monotonic_abstime(int64_t target_ns,
                                          const std::string& stop_file_path,
                                          const volatile sig_atomic_t* signal_flag) {
    constexpr int64_t kStopPollSliceNs = 15LL * kNsPerMs;
    while (true) {
        if ((signal_flag != nullptr && *signal_flag != 0) || file_exists_fast(stop_file_path)) {
            return false;
        }
        const int64_t now_ns = read_clock_ns(CLOCK_MONOTONIC);
        if (now_ns >= target_ns) return true;

        int64_t wake_ns = target_ns;
        if (!stop_file_path.empty() && (target_ns - now_ns) > kStopPollSliceNs) {
            wake_ns = now_ns + kStopPollSliceNs;
        }
        const struct timespec wake_ts = ns_to_timespec(wake_ns);
        const int rc = ::clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &wake_ts, nullptr);
        if (rc == 0) {
            if (wake_ns >= target_ns) return true;
            continue;
        }
        if (rc == EINTR) {
            if (signal_flag != nullptr && *signal_flag != 0) return false;
            continue;
        }
        throw std::runtime_error("clock_nanosleep failed: " + std::string(std::strerror(rc)));
    }
}

inline SampleOutcome execute_isolated_probe(const ProbeSpec& spec,
                                            uint64_t sample_index,
                                            int64_t scheduled_ns,
                                            bool is_final_bracket) {
    if (spec.argv.empty() && spec.shell_cmd.empty()) {
        throw std::invalid_argument("ProbeSpec requires argv or shell_cmd");
    }

    int pipe_fds[2] = {-1, -1};
    if (::pipe(pipe_fds) != 0) {
        throw std::runtime_error("pipe failed: " + std::string(std::strerror(errno)));
    }
    (void)::fcntl(pipe_fds[0], F_SETFD, FD_CLOEXEC);
    (void)::fcntl(pipe_fds[1], F_SETFD, FD_CLOEXEC);

    SampleOutcome outcome{};
    outcome.sample_index = sample_index;
    outcome.scheduled_ns = scheduled_ns;
    outcome.is_final_bracket = is_final_bracket;
    outcome.dispatch_ns = read_clock_ns(CLOCK_MONOTONIC);
    outcome.boottime_ns = read_clock_ns(CLOCK_BOOTTIME);
    outcome.realtime_ns = read_clock_ns(CLOCK_REALTIME);

    const pid_t child_pid = ::fork();
    if (child_pid < 0) {
        const int err = errno;
        (void)::close(pipe_fds[0]);
        (void)::close(pipe_fds[1]);
        throw std::runtime_error("fork failed: " + std::string(std::strerror(err)));
    }

    if (child_pid == 0) {
        (void)::setpgid(0, 0);
        struct sigaction default_sa {};
        default_sa.sa_handler = SIG_DFL;
        sigemptyset(&default_sa.sa_mask);
        (void)::sigaction(SIGINT, &default_sa, nullptr);
        (void)::sigaction(SIGTERM, &default_sa, nullptr);
        (void)::sigaction(SIGPIPE, &default_sa, nullptr);
        sigset_t empty_mask;
        sigemptyset(&empty_mask);
        (void)::sigprocmask(SIG_SETMASK, &empty_mask, nullptr);

        const int devnull = ::open("/dev/null", O_RDWR);
        if (devnull >= 0) {
            (void)::dup2(devnull, STDIN_FILENO);
            if (!spec.merge_stderr) (void)::dup2(devnull, STDERR_FILENO);
            if (devnull > STDERR_FILENO) (void)::close(devnull);
        }
        if (::dup2(pipe_fds[1], STDOUT_FILENO) < 0) ::_exit(126);
        if (spec.merge_stderr && ::dup2(pipe_fds[1], STDERR_FILENO) < 0) ::_exit(126);
        (void)::close(pipe_fds[0]);
        (void)::close(pipe_fds[1]);

        if (!spec.shell_cmd.empty()) {
            ::execl("/bin/sh", "sh", "-c", spec.shell_cmd.c_str(), static_cast<char*>(nullptr));
            ::_exit(127);
        }
        std::vector<char*> c_argv;
        c_argv.reserve(spec.argv.size() + 1U);
        for (const auto& arg : spec.argv) c_argv.push_back(const_cast<char*>(arg.c_str()));
        c_argv.push_back(nullptr);
        ::execvp(c_argv[0], c_argv.data());
        ::_exit(127);
    }

    (void)::setpgid(child_pid, child_pid);
    (void)::close(pipe_fds[1]);
    const int rflags = ::fcntl(pipe_fds[0], F_GETFL, 0);
    if (rflags >= 0) (void)::fcntl(pipe_fds[0], F_SETFL, rflags | O_NONBLOCK);

    const bool has_deadline = spec.timeout_ns > 0;
    const int64_t deadline_ns = has_deadline ? (outcome.dispatch_ns + spec.timeout_ns) : 0;
    bool pipe_open = true;
    bool child_reaped = false;
    int wait_status = 0;
    char chunk[4096];

    auto append_stdout = [&](const char* buf, size_t n) {
        if (outcome.stdout_payload.size() < spec.max_stdout_bytes) {
            const size_t room = spec.max_stdout_bytes - outcome.stdout_payload.size();
            outcome.stdout_payload.append(buf, std::min(n, room));
        }
    };

    while (pipe_open || !child_reaped) {
        const int64_t now_ns = read_clock_ns(CLOCK_MONOTONIC);
        if (has_deadline && now_ns >= deadline_ns) {
            outcome.timed_out = true;
            (void)::kill(-child_pid, SIGKILL);
            (void)::kill(child_pid, SIGKILL);
            break;
        }
        int wait_ms = pipe_open ? 25 : 5;
        if (has_deadline) {
            const int64_t ceil_ms = (deadline_ns - now_ns + kNsPerMs - 1LL) / kNsPerMs;
            wait_ms = static_cast<int>(std::min<int64_t>(ceil_ms, wait_ms));
        }

        if (pipe_open) {
            struct pollfd pfd {};
            pfd.fd = pipe_fds[0];
            pfd.events = POLLIN;
            if (::poll(&pfd, 1, wait_ms) > 0 && (pfd.revents & (POLLIN | POLLHUP | POLLERR)) != 0) {
                while (true) {
                    const ssize_t n = ::read(pipe_fds[0], chunk, sizeof(chunk));
                    if (n > 0) {
                        append_stdout(chunk, static_cast<size_t>(n));
                    } else if (n == 0) {
                        pipe_open = false;
                        break;
                    } else if (errno == EINTR) {
                        continue;
                    } else {
                        if (errno != EAGAIN && errno != EWOULDBLOCK) pipe_open = false;
                        break;
                    }
                }
            }
        } else {
            const struct timespec ts = ns_to_timespec(static_cast<int64_t>(std::max(1, wait_ms)) * kNsPerMs);
            (void)::nanosleep(&ts, nullptr);
        }

        if (!child_reaped && ::waitpid(child_pid, &wait_status, WNOHANG) == child_pid) {
            child_reaped = true;
        }
    }

    while (!outcome.timed_out) {
        const ssize_t n = ::read(pipe_fds[0], chunk, sizeof(chunk));
        if (n > 0) append_stdout(chunk, static_cast<size_t>(n));
        else break;
    }
    (void)::close(pipe_fds[0]);

    while (!child_reaped) {
        const pid_t w = ::waitpid(child_pid, &wait_status, 0);
        if (w == child_pid || (w < 0 && errno != EINTR)) break;
    }
    (void)::kill(-child_pid, SIGKILL);

    outcome.end_ns = read_clock_ns(CLOCK_MONOTONIC);
    if (WIFEXITED(wait_status)) outcome.rc = WEXITSTATUS(wait_status);
    else if (WIFSIGNALED(wait_status)) outcome.rc = -WTERMSIG(wait_status);
    else outcome.rc = -1;
    if (outcome.timed_out && outcome.rc == 0) outcome.rc = -SIGKILL;
    return outcome;
}

inline std::string format_framed_sample(const SampleOutcome& sample) {
    std::ostringstream oss;
    oss << "__SAMPLE_BEGIN__\n__TS_BEGIN_NS__=" << sample.dispatch_ns << "\n" << sample.stdout_payload;
    if (!sample.stdout_payload.empty() && sample.stdout_payload.back() != '\n') oss << "\n";
    oss << "__TS_END_NS__=" << sample.end_ns << "\n__SAMPLE_END__\n";
    return oss.str();
}

inline std::string format_schedule_csv_header() {
    return "scheduled_ns,dispatch_ns,end_ns,rc,boottime_ns,realtime_ns\n";
}

inline std::string format_schedule_csv_row(const SampleOutcome& sample) {
    std::ostringstream oss;
    oss << sample.scheduled_ns << "," << sample.dispatch_ns << "," << sample.end_ns << ","
        << sample.rc << "," << sample.boottime_ns << "," << sample.realtime_ns << "\n";
    return oss.str();
}

inline JitterSummary compute_jitter_summary(const std::vector<int64_t>& lags_ns) {
    JitterSummary s{};
    if (lags_ns.empty()) return s;
    std::vector<int64_t> sorted = lags_ns;
    std::sort(sorted.begin(), sorted.end());
    s.count = sorted.size();
    s.min_lag_ns = sorted.front();
    s.max_lag_ns = sorted.back();
    long double acc = 0.0L;
    for (int64_t v : sorted) acc += static_cast<long double>(v);
    s.mean_lag_ns = static_cast<double>(acc / sorted.size());
    auto pct = [&](double q) -> int64_t {
        const size_t idx = static_cast<size_t>(std::llround(q * static_cast<double>(sorted.size() - 1U)));
        return sorted[std::min(idx, sorted.size() - 1U)];
    };
    s.p50_lag_ns = pct(0.50);
    s.p95_lag_ns = pct(0.95);
    s.p99_lag_ns = pct(0.99);
    return s;
}

inline const char* stop_reason_name(StopReason reason) noexcept {
    switch (reason) {
        case StopReason::kNone: return "none";
        case StopReason::kMaxSamples: return "max_samples";
        case StopReason::kDurationLimit: return "duration_limit";
        case StopReason::kStopFile: return "stop_file";
        case StopReason::kSignal: return "signal";
        case StopReason::kFatalError: return "fatal_error";
    }
    return "unknown";
}

}  // namespace periodic_sampler

#endif  // PERIODIC_SAMPLER_HPP_
