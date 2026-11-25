#include "periodic_sampler.hpp"

#include <csignal>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

volatile sig_atomic_t g_received_signal = 0;

void handle_shutdown_signal(int signo) noexcept {
    if (g_received_signal == 0) g_received_signal = signo;
}

void install_signal_handlers() {
    struct sigaction sa {};
    sa.sa_handler = handle_shutdown_signal;
    sigemptyset(&sa.sa_mask);
    if (::sigaction(SIGINT, &sa, nullptr) != 0 || ::sigaction(SIGTERM, &sa, nullptr) != 0) {
        throw std::runtime_error("sigaction installation failed");
    }
    struct sigaction ign {};
    ign.sa_handler = SIG_IGN;
    sigemptyset(&ign.sa_mask);
    (void)::sigaction(SIGPIPE, &ign, nullptr);
}

std::string json_escape(const std::string& input) {
    std::ostringstream oss;
    for (unsigned char ch : input) {
        switch (ch) {
            case '"': oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default:
                if (ch < 0x20U) {
                    oss << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<unsigned int>(ch) << std::dec;
                } else {
                    oss << static_cast<char>(ch);
                }
        }
    }
    return oss.str();
}

std::string format_meta_json(const periodic_sampler::SamplerConfig& cfg,
                             const periodic_sampler::RunSummary& s) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3) << "{\n"
        << "  \"succeeded\": " << (s.succeeded ? "true" : "false")
        << ", \"stop_reason\": \"" << periodic_sampler::stop_reason_name(s.stop_reason)
        << "\", \"terminating_signal\": " << s.terminating_signal
        << ", \"interval_ns\": " << cfg.interval_ns
        << ", \"probe_timeout_ns\": " << cfg.probe.timeout_ns << ",\n"
        << "  \"timerslack\": {\"modified\": " << (cfg.set_timerslack ? "true" : "false")
        << ", \"previous_ns\": " << s.timerslack.previous_ns
        << ", \"requested_ns\": " << s.timerslack.requested_ns
        << ", \"effective_ns\": " << s.timerslack.effective_ns
        << ", \"errno\": " << s.timerslack.error_number << "},\n"
        << "  \"counts\": {\"total_samples\": " << s.total_samples
        << ", \"periodic_samples\": " << s.periodic_samples
        << ", \"nonzero_rc_samples\": " << s.nonzero_rc_samples
        << ", \"timed_out_samples\": " << s.timed_out_samples
        << ", \"skipped_ticks\": " << s.skipped_ticks
        << ", \"final_bracket_captured\": " << (s.final_bracket_captured ? "true" : "false") << "},\n"
        << "  \"run_window_ns\": {\"start_ns\": " << s.run_start_ns
        << ", \"end_ns\": " << s.run_end_ns
        << ", \"span_ns\": " << (s.run_end_ns - s.run_start_ns) << "},\n"
        << "  \"dispatch_jitter_ns\": {\"count\": " << s.jitter.count
        << ", \"mean_ns\": " << s.jitter.mean_lag_ns
        << ", \"min_ns\": " << s.jitter.min_lag_ns
        << ", \"p50_ns\": " << s.jitter.p50_lag_ns
        << ", \"p95_ns\": " << s.jitter.p95_lag_ns
        << ", \"p99_ns\": " << s.jitter.p99_lag_ns
        << ", \"max_ns\": " << s.jitter.max_lag_ns << "},\n"
        << "  \"error_detail\": \"" << json_escape(s.error_detail) << "\"\n}\n";
    return oss.str();
}

int64_t parse_positive_i64(const std::string& flag, const std::string& raw) {
    char* end = nullptr;
    errno = 0;
    const long long val = std::strtoll(raw.c_str(), &end, 10);
    if (errno != 0 || end == raw.c_str() || *end != '\0' || val <= 0) {
        throw std::invalid_argument("Flag " + flag + " requires positive integer: " + raw);
    }
    return static_cast<int64_t>(val);
}

uint64_t parse_nonneg_u64(const std::string& flag, const std::string& raw) {
    if (!raw.empty() && raw.front() == '-') {
        throw std::invalid_argument("Flag " + flag + " requires non-negative integer: " + raw);
    }
    char* end = nullptr;
    errno = 0;
    const unsigned long long val = std::strtoull(raw.c_str(), &end, 10);
    if (errno != 0 || end == raw.c_str() || *end != '\0') {
        throw std::invalid_argument("Flag " + flag + " requires non-negative integer: " + raw);
    }
    return static_cast<uint64_t>(val);
}

double parse_positive_double(const std::string& flag, const std::string& raw) {
    char* end = nullptr;
    errno = 0;
    const double val = std::strtod(raw.c_str(), &end);
    if (errno != 0 || end == raw.c_str() || *end != '\0' || !std::isfinite(val) || val <= 0.0) {
        throw std::invalid_argument("Flag " + flag + " requires positive number: " + raw);
    }
    return val;
}

void print_usage(std::ostream& out, const char* prog) {
    out << "Usage: " << prog << " [options] [-- <cmd> [args...]]\n"
        << "  --output <path>            Framed sample stream output path (exclusive)\n"
        << "  --csv-output <path>        Dispatch schedule CSV path (exclusive)\n"
        << "  --schedule-csv <path>      Alias for --csv-output\n"
        << "  --meta-output <path>       Run diagnostic JSON path (exclusive)\n"
        << "  --done-file <path>         Completion status file ('complete'|'failed', exclusive)\n"
        << "  --stop-file <path>         Graceful stop sentinel path\n"
        << "  --interval-ms <num>        Sampling cadence in milliseconds (default: 100)\n"
        << "  --interval-ns <int>        Sampling cadence in nanoseconds\n"
        << "  --probe-timeout-ms <num>   Per-probe monotonic timeout in ms (default: 5000)\n"
        << "  --max-samples <int>        Stop after N periodic samples (0 = unbounded)\n"
        << "  --duration-ms <num>        Stop after total elapsed milliseconds\n"
        << "  --timerslack-ns <int>      Configure Linux prctl(PR_SET_TIMERSLACK)\n"
        << "  --overrun-policy <mode>    'catchup' (default) or 'skip'\n"
        << "  --no-final-bracket         Disable final bracket sample on stop-file/signal\n"
        << "  --fail-on-probe-error      Mark run failed if any probe exits non-zero/times out\n"
        << "  --merge-stderr             Capture probe stderr alongside stdout\n"
        << "  --shell-cmd <string>       Execute command string via /bin/sh -c\n";
}

periodic_sampler::SamplerConfig parse_cli(int argc, char** argv, bool& help_requested) {
    periodic_sampler::SamplerConfig cfg{};
    help_requested = false;
    auto next_arg = [&](int& idx, const std::string& flag) -> std::string {
        if (idx + 1 >= argc) throw std::invalid_argument("Missing argument for " + flag);
        return std::string(argv[++idx]);
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--") {
            for (int j = i + 1; j < argc; ++j) cfg.probe.argv.emplace_back(argv[j]);
            break;
        }
        if (arg == "--help" || arg == "-h") {
            help_requested = true;
            return cfg;
        }
        if (arg == "--output") cfg.output_path = next_arg(i, arg);
        else if (arg == "--csv-output" || arg == "--schedule-csv") cfg.csv_path = next_arg(i, arg);
        else if (arg == "--meta-output") cfg.meta_path = next_arg(i, arg);
        else if (arg == "--done-file") cfg.done_file_path = next_arg(i, arg);
        else if (arg == "--stop-file") cfg.stop_file_path = next_arg(i, arg);
        else if (arg == "--interval-ms") {
            const double ms = parse_positive_double(arg, next_arg(i, arg));
            cfg.interval_ns = std::max<int64_t>(1, std::llround(ms * periodic_sampler::kNsPerMs));
        } else if (arg == "--interval-ns") {
            cfg.interval_ns = parse_positive_i64(arg, next_arg(i, arg));
        } else if (arg == "--probe-timeout-ms") {
            const double ms = parse_positive_double(arg, next_arg(i, arg));
            cfg.probe.timeout_ns = std::max<int64_t>(1, std::llround(ms * periodic_sampler::kNsPerMs));
        } else if (arg == "--max-samples") {
            cfg.max_samples = parse_nonneg_u64(arg, next_arg(i, arg));
        } else if (arg == "--duration-ms") {
            const double ms = parse_positive_double(arg, next_arg(i, arg));
            cfg.max_duration_ns = std::max<int64_t>(1, std::llround(ms * periodic_sampler::kNsPerMs));
        } else if (arg == "--timerslack-ns") {
            cfg.timerslack_ns = parse_nonneg_u64(arg, next_arg(i, arg));
            cfg.set_timerslack = true;
        } else if (arg == "--overrun-policy") {
            const std::string mode = next_arg(i, arg);
            if (mode == "catchup") cfg.overrun_policy = periodic_sampler::OverrunPolicy::kCatchUp;
            else if (mode == "skip") cfg.overrun_policy = periodic_sampler::OverrunPolicy::kSkipMissed;
            else throw std::invalid_argument("Unsupported --overrun-policy: " + mode);
        } else if (arg == "--no-final-bracket") cfg.capture_final_on_stop = false;
        else if (arg == "--fail-on-probe-error") cfg.fail_on_probe_error = true;
        else if (arg == "--merge-stderr") cfg.probe.merge_stderr = true;
        else if (arg == "--shell-cmd" || arg == "--cmd") cfg.probe.shell_cmd = next_arg(i, arg);
        else throw std::invalid_argument("Unrecognized option: " + arg);
    }

    if (cfg.output_path.empty()) throw std::invalid_argument("--output <path> is required");
    if (cfg.probe.argv.empty() == cfg.probe.shell_cmd.empty()) {
        throw std::invalid_argument("Specify exactly one of '-- <cmd> ...' or '--shell-cmd <str>'");
    }
    if (!cfg.done_file_path.empty() && periodic_sampler::file_exists_fast(cfg.done_file_path)) {
        throw std::runtime_error("exclusive open rejected for '" + cfg.done_file_path + "': File exists");
    }
    return cfg;
}

void write_done_marker_exclusive(const std::string& done_path, bool ok) {
    if (done_path.empty()) return;
    periodic_sampler::ExclusiveFileGuard done_guard(done_path, 0644);
    done_guard.write_str(ok ? "complete\n" : "failed\n");
    done_guard.sync_and_close();
}

}  // namespace

int main(int argc, char** argv) {
    using namespace periodic_sampler;
    SamplerConfig cfg{};
    bool help_requested = false;
    try {
        cfg = parse_cli(argc, argv, help_requested);
    } catch (const std::exception& ex) {
        std::cerr << "periodic_cmd_sampler error: " << ex.what() << "\n";
        return 2;
    }
    if (help_requested) {
        print_usage(std::cout, argv[0]);
        return 0;
    }

    RunSummary summary{};
    ExclusiveFileGuard out_guard, csv_guard, meta_guard;
    try {
        out_guard.open_exclusive(cfg.output_path, 0644);
        if (!cfg.csv_path.empty()) {
            csv_guard.open_exclusive(cfg.csv_path, 0644);
            csv_guard.write_str(format_schedule_csv_header());
        }
        if (!cfg.meta_path.empty()) meta_guard.open_exclusive(cfg.meta_path, 0644);
    } catch (const std::exception& ex) {
        std::cerr << "periodic_cmd_sampler output setup error: " << ex.what() << "\n";
        return 2;
    }

    try {
        install_signal_handlers();
        summary.timerslack = apply_timerslack(cfg.set_timerslack, cfg.timerslack_ns);
        summary.run_start_ns = read_clock_ns(CLOCK_MONOTONIC);
        const int64_t base_ns = summary.run_start_ns;
        const int64_t end_deadline_ns = cfg.max_duration_ns > 0 ? (base_ns + cfg.max_duration_ns) : 0;
        std::vector<int64_t> periodic_lags_ns;
        uint64_t tick_idx = 0;

        auto record_sample = [&](const SampleOutcome& sample) {
            out_guard.write_str(format_framed_sample(sample));
            if (csv_guard.is_open()) csv_guard.write_str(format_schedule_csv_row(sample));
            ++summary.total_samples;
            if (!sample.is_final_bracket) {
                ++summary.periodic_samples;
                periodic_lags_ns.push_back(sample.dispatch_ns - sample.scheduled_ns);
            } else {
                summary.final_bracket_captured = true;
            }
            if (sample.rc != 0) ++summary.nonzero_rc_samples;
            if (sample.timed_out) ++summary.timed_out_samples;
        };

        while (true) {
            if (g_received_signal != 0) {
                summary.stop_reason = StopReason::kSignal;
                summary.terminating_signal = static_cast<int>(g_received_signal);
                break;
            }
            if (file_exists_fast(cfg.stop_file_path)) { summary.stop_reason = StopReason::kStopFile; break; }
            if (cfg.max_samples > 0 && summary.periodic_samples >= cfg.max_samples) {
                summary.stop_reason = StopReason::kMaxSamples;
                break;
            }
            int64_t scheduled_ns = base_ns + static_cast<int64_t>(tick_idx) * cfg.interval_ns;
            const int64_t now_before = read_clock_ns(CLOCK_MONOTONIC);
            if (end_deadline_ns > 0 && scheduled_ns > end_deadline_ns && summary.periodic_samples > 0) {
                summary.stop_reason = StopReason::kDurationLimit;
                break;
            }
            if (cfg.overrun_policy == OverrunPolicy::kSkipMissed && now_before > scheduled_ns && tick_idx > 0) {
                const uint64_t next_tick = static_cast<uint64_t>((now_before - base_ns) / cfg.interval_ns) + 1ULL;
                if (next_tick > tick_idx) {
                    summary.skipped_ticks += (next_tick - tick_idx);
                    tick_idx = next_tick;
                    scheduled_ns = base_ns + static_cast<int64_t>(tick_idx) * cfg.interval_ns;
                }
            }
            if (now_before < scheduled_ns &&
                !sleep_until_monotonic_abstime(scheduled_ns, cfg.stop_file_path, &g_received_signal)) {
                if (g_received_signal != 0) {
                    summary.stop_reason = StopReason::kSignal;
                    summary.terminating_signal = static_cast<int>(g_received_signal);
                } else {
                    summary.stop_reason = StopReason::kStopFile;
                }
                break;
            }
            if (end_deadline_ns > 0 && read_clock_ns(CLOCK_MONOTONIC) > end_deadline_ns && summary.periodic_samples > 0) {
                summary.stop_reason = StopReason::kDurationLimit;
                break;
            }
            record_sample(execute_isolated_probe(cfg.probe, summary.total_samples, scheduled_ns, false));
            ++tick_idx;
        }

        if ((summary.stop_reason == StopReason::kStopFile || summary.stop_reason == StopReason::kSignal) &&
            cfg.capture_final_on_stop) {
            const int64_t bracket_ns = read_clock_ns(CLOCK_MONOTONIC);
            record_sample(execute_isolated_probe(cfg.probe, summary.total_samples, bracket_ns, true));
        }

        summary.run_end_ns = read_clock_ns(CLOCK_MONOTONIC);
        summary.jitter = compute_jitter_summary(periodic_lags_ns);
        summary.succeeded = (summary.total_samples > 0) &&
                            (!cfg.fail_on_probe_error || summary.nonzero_rc_samples == 0);
        if (!summary.succeeded && summary.error_detail.empty()) {
            summary.error_detail = summary.total_samples == 0 ? "zero samples captured" : "probe error observed";
        }
        out_guard.sync_and_close();
        if (csv_guard.is_open()) csv_guard.sync_and_close();
        if (meta_guard.is_open()) {
            meta_guard.write_str(format_meta_json(cfg, summary));
            meta_guard.sync_and_close();
        }
        write_done_marker_exclusive(cfg.done_file_path, summary.succeeded);
    } catch (const std::exception& ex) {
        summary.stop_reason = StopReason::kFatalError;
        summary.error_detail = ex.what();
        summary.run_end_ns = read_clock_ns(CLOCK_MONOTONIC);
        try {
            if (meta_guard.is_open()) {
                meta_guard.write_str(format_meta_json(cfg, summary));
                meta_guard.sync_and_close();
            }
            write_done_marker_exclusive(cfg.done_file_path, false);
        } catch (...) {}
        std::cerr << "periodic_cmd_sampler fatal error: " << ex.what() << "\n";
        return 1;
    }
    return summary.succeeded ? 0 : 1;
}
