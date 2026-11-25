#ifndef POWER_WINDOW_HPP_
#define POWER_WINDOW_HPP_

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace power_window {

constexpr int64_t kNsPerMs = 1000000LL;
constexpr long double kSecPerNs = 1.0e-9L;

struct PowerSample {
    int64_t timestamp_ns = 0;
    double voltage_v = 0.0;
    double current_a = 0.0;
    bool valid = true;

    double power_w() const noexcept { return voltage_v * current_a; }
};

struct WorkloadWindow {
    std::string request_id;
    int64_t start_ns = 0;
    int64_t end_ns = 0;
    uint64_t output_tokens = 0;
};

struct IntegrationConfig {
    int64_t max_gap_ns = 50LL * kNsPerMs;
    int64_t idle_lead_ns = 0;
    int64_t idle_trail_ns = 0;
    int64_t idle_guard_ns = 0;
    bool has_fixed_idle_w = false;
    double fixed_idle_w = 0.0;
};

enum class RejectCode {
    kOk,
    kInvalidWindowBounds,
    kEmptyTelemetry,
    kMissingLeftBracket,
    kMissingRightBracket,
    kInvalidSampleInSpan,
    kExcessiveSampleGap,
    kIdleWindowInvalid
};

struct IntervalEnergy {
    RejectCode status = RejectCode::kOk;
    std::string detail;
    double duration_s = 0.0;
    double trapezoid_j = 0.0;
    double midpoint_j = 0.0;
    double mean_trapezoid_w = 0.0;
    double mean_midpoint_w = 0.0;
    int64_t max_observed_gap_ns = 0;
    size_t spanned_samples = 0;
};

struct AlignedWindowResult {
    std::string request_id;
    int64_t start_ns = 0;
    int64_t end_ns = 0;
    uint64_t output_tokens = 0;
    RejectCode status = RejectCode::kOk;
    std::string reject_detail;
    double duration_s = 0.0;
    int64_t max_observed_gap_ns = 0;
    size_t spanned_samples = 0;
    double gross_trapezoid_j = 0.0;
    double gross_midpoint_j = 0.0;
    double mean_gross_trapezoid_w = 0.0;
    double mean_gross_midpoint_w = 0.0;
    bool baseline_applied = false;
    double idle_baseline_trapezoid_w = 0.0;
    double idle_baseline_midpoint_w = 0.0;
    double idle_energy_trapezoid_j = 0.0;
    double idle_energy_midpoint_j = 0.0;
    double net_trapezoid_j = 0.0;
    double net_midpoint_j = 0.0;
    double gross_trapezoid_j_per_token = 0.0;
    double gross_midpoint_j_per_token = 0.0;
    double net_trapezoid_j_per_token = 0.0;
    double net_midpoint_j_per_token = 0.0;
};

struct AggregateSummary {
    size_t total_windows = 0;
    size_t accepted_windows = 0;
    size_t rejected_windows = 0;
    uint64_t total_output_tokens = 0;
    double total_duration_s = 0.0;
    double total_gross_trapezoid_j = 0.0;
    double total_gross_midpoint_j = 0.0;
    double total_net_trapezoid_j = 0.0;
    double total_net_midpoint_j = 0.0;
    double mean_gross_trapezoid_w = 0.0;
    double mean_net_trapezoid_w = 0.0;
    double aggregate_gross_trapezoid_j_per_token = 0.0;
    double aggregate_net_trapezoid_j_per_token = 0.0;
    double aggregate_net_midpoint_j_per_token = 0.0;
};

inline const char* reject_code_name(RejectCode code) noexcept {
    switch (code) {
        case RejectCode::kOk: return "ok";
        case RejectCode::kInvalidWindowBounds: return "invalid_window_bounds";
        case RejectCode::kEmptyTelemetry: return "empty_telemetry";
        case RejectCode::kMissingLeftBracket: return "missing_left_bracket";
        case RejectCode::kMissingRightBracket: return "missing_right_bracket";
        case RejectCode::kInvalidSampleInSpan: return "invalid_sample_in_span";
        case RejectCode::kExcessiveSampleGap: return "excessive_sample_gap";
        case RejectCode::kIdleWindowInvalid: return "idle_window_invalid";
    }
    return "unknown";
}

inline std::string trim_str(const std::string& s) {
    size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1U]))) --e;
    return s.substr(b, e - b);
}

inline void write_exclusive_output(const std::string& path, const std::string& payload) {
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (fd < 0) {
        throw std::runtime_error("exclusive open rejected for '" + path + "': " +
                                 std::string(std::strerror(errno)));
    }
    size_t off = 0;
    while (off < payload.size()) {
        const ssize_t n = ::write(fd, payload.data() + off, payload.size() - off);
        if (n < 0) {
            if (errno == EINTR) continue;
            const int err = errno;
            (void)::close(fd);
            throw std::runtime_error("write failed for '" + path + "': " + std::string(std::strerror(err)));
        }
        off += static_cast<size_t>(n);
    }
    (void)::fsync(fd);
    if (::close(fd) != 0) {
        throw std::runtime_error("close failed for '" + path + "'");
    }
}

inline std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> cols;
    std::string cur;
    bool in_quotes = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '"') {
            if (in_quotes && i + 1U < line.size() && line[i + 1U] == '"') {
                cur.push_back('"');
                ++i;
            } else {
                in_quotes = !in_quotes;
            }
        } else if (c == ',' && !in_quotes) {
            cols.push_back(trim_str(cur));
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    if (in_quotes) throw std::runtime_error("Unterminated quote in CSV row");
    cols.push_back(trim_str(cur));
    return cols;
}

inline std::map<std::string, std::string> parse_flat_json_object(const std::string& line) {
    const std::string s = trim_str(line);
    if (s.size() < 2U || s.front() != '{' || s.back() != '}') {
        throw std::runtime_error("Malformed JSONL object: " + s);
    }
    std::map<std::string, std::string> kv;
    size_t pos = 1U;
    const size_t limit = s.size() - 1U;
    auto skip_ws = [&]() {
        while (pos < limit && std::isspace(static_cast<unsigned char>(s[pos]))) ++pos;
    };
    auto read_quoted = [&]() -> std::string {
        if (pos >= limit || s[pos] != '"') throw std::runtime_error("Expected JSON string");
        ++pos;
        std::string out;
        while (pos < limit) {
            const char c = s[pos++];
            if (c == '"') return out;
            if (c == '\\' && pos < limit) out.push_back(s[pos++]);
            else out.push_back(c);
        }
        throw std::runtime_error("Unterminated JSON string");
    };

    skip_ws();
    while (pos < limit) {
        const std::string key = read_quoted();
        skip_ws();
        if (pos >= limit || s[pos] != ':') throw std::runtime_error("Expected ':' in JSONL");
        ++pos;
        skip_ws();
        std::string val;
        if (pos < limit && s[pos] == '"') {
            val = read_quoted();
        } else {
            const size_t start = pos;
            while (pos < limit && s[pos] != ',') ++pos;
            val = trim_str(s.substr(start, pos - start));
        }
        kv[key] = val;
        skip_ws();
        if (pos < limit) {
            if (s[pos] != ',') throw std::runtime_error("Expected ',' in JSONL");
            ++pos;
            skip_ws();
        }
    }
    return kv;
}

inline int64_t parse_i64_field(const std::string& raw, const std::string& field) {
    const std::string s = trim_str(raw);
    char* end = nullptr;
    errno = 0;
    const long long v = std::strtoll(s.c_str(), &end, 10);
    if (errno != 0 || end == s.c_str() || *end != '\0') {
        throw std::runtime_error("Invalid integer for " + field + ": '" + s + "'");
    }
    return static_cast<int64_t>(v);
}

inline uint64_t parse_u64_field(const std::string& raw, const std::string& field) {
    const std::string s = trim_str(raw);
    if (s.empty() || s.front() == '-') {
        throw std::runtime_error("Negative unsigned field " + field + ": '" + s + "'");
    }
    char* end = nullptr;
    errno = 0;
    const unsigned long long v = std::strtoull(s.c_str(), &end, 10);
    if (errno != 0 || end == s.c_str() || *end != '\0') {
        throw std::runtime_error("Invalid unsigned integer for " + field + ": '" + s + "'");
    }
    return static_cast<uint64_t>(v);
}

inline double parse_double_field(const std::string& raw, const std::string& field) {
    const std::string s = trim_str(raw);
    char* end = nullptr;
    errno = 0;
    const double v = std::strtod(s.c_str(), &end);
    if (errno != 0 || end == s.c_str() || *end != '\0' || !std::isfinite(v)) {
        throw std::runtime_error("Invalid floating-point value for " + field + ": '" + s + "'");
    }
    return v;
}

inline bool parse_bool_field(const std::string& raw, const std::string& field) {
    std::string lower = trim_str(raw);
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lower == "1" || lower == "true") return true;
    if (lower == "0" || lower == "false") return false;
    throw std::runtime_error("Invalid boolean for " + field + ": '" + raw + "'");
}

inline void validate_sorted_telemetry(const std::vector<PowerSample>& samples) {
    for (size_t i = 0; i < samples.size(); ++i) {
        if (!std::isfinite(samples[i].voltage_v) || !std::isfinite(samples[i].current_a) ||
            samples[i].voltage_v < 0.0) {
            throw std::runtime_error("Invalid telemetry voltage/current at index " + std::to_string(i));
        }
        if (i > 0 && samples[i].timestamp_ns <= samples[i - 1U].timestamp_ns) {
            throw std::runtime_error("Non-increasing telemetry timestamp_ns at index " + std::to_string(i));
        }
    }
}

inline IntervalEnergy integrate_interval(const std::vector<PowerSample>& samples,
                                         int64_t start_ns,
                                         int64_t end_ns,
                                         int64_t max_gap_ns) {
    IntervalEnergy out{};
    if (end_ns <= start_ns) {
        out.status = RejectCode::kInvalidWindowBounds;
        out.detail = "end_ns must exceed start_ns";
        return out;
    }
    if (samples.empty()) {
        out.status = RejectCode::kEmptyTelemetry;
        out.detail = "telemetry sample vector is empty";
        return out;
    }

    auto upper_start = std::upper_bound(
        samples.begin(), samples.end(), start_ns,
        [](int64_t t, const PowerSample& s) { return t < s.timestamp_ns; });
    if (upper_start == samples.begin()) {
        out.status = RejectCode::kMissingLeftBracket;
        out.detail = "no sample at or before start_ns";
        return out;
    }
    const size_t left_idx = static_cast<size_t>(std::distance(samples.begin(), upper_start) - 1);

    auto lower_end = std::lower_bound(
        samples.begin(), samples.end(), end_ns,
        [](const PowerSample& s, int64_t t) { return s.timestamp_ns < t; });
    if (lower_end == samples.end()) {
        out.status = RejectCode::kMissingRightBracket;
        out.detail = "no sample at or after end_ns";
        return out;
    }
    const size_t right_idx = static_cast<size_t>(std::distance(samples.begin(), lower_end));

    for (size_t k = left_idx; k <= right_idx; ++k) {
        if (!samples[k].valid) {
            out.status = RejectCode::kInvalidSampleInSpan;
            out.detail = "invalid sample at timestamp_ns=" + std::to_string(samples[k].timestamp_ns);
            return out;
        }
        if (k > left_idx) {
            const int64_t gap = samples[k].timestamp_ns - samples[k - 1U].timestamp_ns;
            out.max_observed_gap_ns = std::max(out.max_observed_gap_ns, gap);
            if (max_gap_ns > 0 && gap > max_gap_ns) {
                out.status = RejectCode::kExcessiveSampleGap;
                out.detail = "sample gap " + std::to_string(gap) + " ns exceeds max_gap_ns " +
                             std::to_string(max_gap_ns);
                return out;
            }
        }
    }

    long double trap_j = 0.0L;
    long double mid_j = 0.0L;
    for (size_t k = left_idx; k < right_idx; ++k) {
        const int64_t t0 = samples[k].timestamp_ns;
        const int64_t t1 = samples[k + 1U].timestamp_ns;
        const int64_t a = std::max(t0, start_ns);
        const int64_t b = std::min(t1, end_ns);
        if (b <= a) continue;

        const long double span = static_cast<long double>(t1 - t0);
        const long double p0 = static_cast<long double>(samples[k].power_w());
        const long double p1 = static_cast<long double>(samples[k + 1U].power_w());

        const long double pa = p0 + (static_cast<long double>(a - t0) / span) * (p1 - p0);
        const long double pb = p0 + (static_cast<long double>(b - t0) / span) * (p1 - p0);
        const long double dt_s = static_cast<long double>(b - a) * kSecPerNs;
        trap_j += 0.5L * (pa + pb) * dt_s;

        const long double mid_ns = 0.5L * (static_cast<long double>(t0) + static_cast<long double>(t1));
        const long double a_ld = static_cast<long double>(a);
        const long double b_ld = static_cast<long double>(b);
        const long double left_ns = std::max(0.0L, std::min(b_ld, mid_ns) - a_ld);
        const long double right_ns = std::max(0.0L, b_ld - std::max(a_ld, mid_ns));
        mid_j += (p0 * left_ns + p1 * right_ns) * kSecPerNs;
    }

    out.duration_s = static_cast<double>(static_cast<long double>(end_ns - start_ns) * kSecPerNs);
    out.trapezoid_j = static_cast<double>(trap_j);
    out.midpoint_j = static_cast<double>(mid_j);
    out.mean_trapezoid_w = out.duration_s > 0.0 ? (out.trapezoid_j / out.duration_s) : 0.0;
    out.mean_midpoint_w = out.duration_s > 0.0 ? (out.midpoint_j / out.duration_s) : 0.0;
    out.spanned_samples = right_idx - left_idx + 1U;
    return out;
}

inline AlignedWindowResult evaluate_workload_window(const std::vector<PowerSample>& samples,
                                                    const WorkloadWindow& win,
                                                    const IntegrationConfig& cfg) {
    AlignedWindowResult res{};
    res.request_id = win.request_id;
    res.start_ns = win.start_ns;
    res.end_ns = win.end_ns;
    res.output_tokens = win.output_tokens;

    const IntervalEnergy main_ie = integrate_interval(samples, win.start_ns, win.end_ns, cfg.max_gap_ns);
    if (main_ie.status != RejectCode::kOk) {
        res.status = main_ie.status;
        res.reject_detail = main_ie.detail;
        return res;
    }

    res.duration_s = main_ie.duration_s;
    res.max_observed_gap_ns = main_ie.max_observed_gap_ns;
    res.spanned_samples = main_ie.spanned_samples;
    res.gross_trapezoid_j = main_ie.trapezoid_j;
    res.gross_midpoint_j = main_ie.midpoint_j;
    res.mean_gross_trapezoid_w = main_ie.mean_trapezoid_w;
    res.mean_gross_midpoint_w = main_ie.mean_midpoint_w;

    if (cfg.has_fixed_idle_w) {
        res.baseline_applied = true;
        res.idle_baseline_trapezoid_w = cfg.fixed_idle_w;
        res.idle_baseline_midpoint_w = cfg.fixed_idle_w;
    } else if (cfg.idle_lead_ns > 0 || cfg.idle_trail_ns > 0) {
        double idle_trap_j = 0.0;
        double idle_mid_j = 0.0;
        double idle_dur_s = 0.0;
        if (cfg.idle_lead_ns > 0) {
            const int64_t l_end = win.start_ns - cfg.idle_guard_ns;
            const int64_t l_start = l_end - cfg.idle_lead_ns;
            const IntervalEnergy lead = integrate_interval(samples, l_start, l_end, cfg.max_gap_ns);
            if (lead.status != RejectCode::kOk) {
                res.status = RejectCode::kIdleWindowInvalid;
                res.reject_detail = "idle lead window rejected: " + lead.detail;
                return res;
            }
            idle_trap_j += lead.trapezoid_j;
            idle_mid_j += lead.midpoint_j;
            idle_dur_s += lead.duration_s;
        }
        if (cfg.idle_trail_ns > 0) {
            const int64_t r_start = win.end_ns + cfg.idle_guard_ns;
            const int64_t r_end = r_start + cfg.idle_trail_ns;
            const IntervalEnergy trail = integrate_interval(samples, r_start, r_end, cfg.max_gap_ns);
            if (trail.status != RejectCode::kOk) {
                res.status = RejectCode::kIdleWindowInvalid;
                res.reject_detail = "idle trail window rejected: " + trail.detail;
                return res;
            }
            idle_trap_j += trail.trapezoid_j;
            idle_mid_j += trail.midpoint_j;
            idle_dur_s += trail.duration_s;
        }
        res.baseline_applied = true;
        res.idle_baseline_trapezoid_w = idle_trap_j / idle_dur_s;
        res.idle_baseline_midpoint_w = idle_mid_j / idle_dur_s;
    }

    res.idle_energy_trapezoid_j = res.idle_baseline_trapezoid_w * res.duration_s;
    res.idle_energy_midpoint_j = res.idle_baseline_midpoint_w * res.duration_s;
    res.net_trapezoid_j = res.gross_trapezoid_j - res.idle_energy_trapezoid_j;
    res.net_midpoint_j = res.gross_midpoint_j - res.idle_energy_midpoint_j;

    if (res.output_tokens > 0) {
        const double tok = static_cast<double>(res.output_tokens);
        res.gross_trapezoid_j_per_token = res.gross_trapezoid_j / tok;
        res.gross_midpoint_j_per_token = res.gross_midpoint_j / tok;
        res.net_trapezoid_j_per_token = res.net_trapezoid_j / tok;
        res.net_midpoint_j_per_token = res.net_midpoint_j / tok;
    }
    return res;
}

inline AggregateSummary summarize_aligned_windows(const std::vector<AlignedWindowResult>& results) {
    AggregateSummary agg{};
    agg.total_windows = results.size();
    for (const auto& r : results) {
        if (r.status != RejectCode::kOk) {
            ++agg.rejected_windows;
            continue;
        }
        ++agg.accepted_windows;
        agg.total_output_tokens += r.output_tokens;
        agg.total_duration_s += r.duration_s;
        agg.total_gross_trapezoid_j += r.gross_trapezoid_j;
        agg.total_gross_midpoint_j += r.gross_midpoint_j;
        agg.total_net_trapezoid_j += r.net_trapezoid_j;
        agg.total_net_midpoint_j += r.net_midpoint_j;
    }
    if (agg.total_duration_s > 0.0) {
        agg.mean_gross_trapezoid_w = agg.total_gross_trapezoid_j / agg.total_duration_s;
        agg.mean_net_trapezoid_w = agg.total_net_trapezoid_j / agg.total_duration_s;
    }
    if (agg.total_output_tokens > 0) {
        const double tok = static_cast<double>(agg.total_output_tokens);
        agg.aggregate_gross_trapezoid_j_per_token = agg.total_gross_trapezoid_j / tok;
        agg.aggregate_net_trapezoid_j_per_token = agg.total_net_trapezoid_j / tok;
        agg.aggregate_net_midpoint_j_per_token = agg.total_net_midpoint_j / tok;
    }
    return agg;
}

}  // namespace power_window

#endif  // POWER_WINDOW_HPP_
