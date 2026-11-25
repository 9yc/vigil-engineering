#ifndef CPUFREQ_RESIDENCY_HPP_
#define CPUFREQ_RESIDENCY_HPP_

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace cpufreq_residency {

struct StateSample { uint64_t freq_khz = 0; uint64_t ticks = 0; };

struct PolicySnapshot {
    int policy_id = 0;
    std::string policy_name, scaling_governor;
    std::vector<int> related_cpus, affected_cpus;
    uint64_t scaling_cur_freq_khz = 0, scaling_min_freq_khz = 0, scaling_max_freq_khz = 0;
    uint64_t cpuinfo_min_freq_khz = 0, cpuinfo_max_freq_khz = 0, total_trans = 0;
    bool has_total_trans = false;
    std::vector<StateSample> time_in_state;

    uint64_t total_ticks() const noexcept {
        uint64_t sum = 0;
        for (const auto& s : time_in_state) sum += s.ticks;
        return sum;
    }

    double weighted_mean_freq_khz() const noexcept {
        long double weighted = 0.0L;
        uint64_t sum = 0;
        for (const auto& s : time_in_state) {
            weighted += static_cast<long double>(s.freq_khz) * static_cast<long double>(s.ticks);
            sum += s.ticks;
        }
        return sum > 0 ? static_cast<double>(weighted / static_cast<long double>(sum)) : 0.0;
    }

    double cap_ratio() const noexcept {
        return cpuinfo_max_freq_khz > 0
                   ? static_cast<double>(scaling_max_freq_khz) / static_cast<double>(cpuinfo_max_freq_khz)
                   : 0.0;
    }
};

struct SystemSnapshot {
    std::string source_label;
    int64_t captured_monotonic_ns = 0;
    std::vector<PolicySnapshot> policies;
};

struct StateDelta {
    uint64_t freq_khz = 0, before_ticks = 0, after_ticks = 0, delta_ticks = 0;
    double delta_ms = 0.0, share = 0.0;
    bool in_upper_band = false;
};

struct PolicyDelta {
    int policy_id = 0;
    std::string policy_name, before_governor, after_governor;
    std::vector<int> related_cpus;
    uint64_t cpuinfo_min_freq_khz = 0, cpuinfo_max_freq_khz = 0;
    uint64_t before_scaling_max_khz = 0, after_scaling_max_khz = 0;
    double before_cap_ratio = 0.0, after_cap_ratio = 0.0, min_cap_ratio = 0.0;
    bool cap_throttled = false;
    uint64_t total_delta_ticks = 0, upper_band_threshold_khz = 0, delta_transitions = 0;
    double total_delta_ms = 0.0, weighted_avg_freq_khz = 0.0, normalized_freq_ratio = 0.0;
    double upper_band_share = 0.0, transitions_per_sec = 0.0;
    std::vector<StateDelta> histogram;
};

struct ClusterAsymmetry {
    size_t policy_count = 0, active_policy_count = 0;
    double max_normalized_freq_ratio = 0.0, min_normalized_freq_ratio = 0.0, normalized_freq_spread = 0.0;
    double max_upper_band_share = 0.0, min_upper_band_share = 0.0, upper_band_share_spread = 0.0;
    double min_cap_ratio_across_policies = 1.0;
    bool governor_heterogeneous = false;
    std::vector<int> capped_policy_ids;
};

struct DeltaAuditReport {
    double top_fraction = 0.8;
    double tick_ms = 10.0;
    std::vector<PolicyDelta> policies;
    ClusterAsymmetry asymmetry;
};

inline std::string trim_ws(const std::string& s) {
    size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1U]))) --e;
    return s.substr(b, e - b);
}

inline bool is_directory_path(const std::string& path) noexcept {
    struct stat st {};
    return ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

inline bool is_regular_file_path(const std::string& path) noexcept {
    struct stat st {};
    return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

inline std::string read_text_file(const std::string& path) {
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs.is_open()) throw std::runtime_error("Unable to read file '" + path + "'");
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

inline void write_exclusive_file(const std::string& path, const std::string& content) {
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (fd < 0) {
        throw std::runtime_error("exclusive open rejected for '" + path + "': " + std::string(std::strerror(errno)));
    }
    for (size_t off = 0; off < content.size();) {
        const ssize_t n = ::write(fd, content.data() + off, content.size() - off);
        if (n < 0) {
            if (errno == EINTR) continue;
            const int err = errno;
            (void)::close(fd);
            throw std::runtime_error("write failed for '" + path + "': " + std::string(std::strerror(err)));
        }
        off += static_cast<size_t>(n);
    }
    (void)::fsync(fd);
    if (::close(fd) != 0) throw std::runtime_error("close failed for '" + path + "'");
}

inline uint64_t parse_u64_strict(const std::string& raw, const std::string& ctx) {
    const std::string s = trim_ws(raw);
    if (s.empty() || s.front() == '-') throw std::runtime_error("Invalid unsigned integer in " + ctx);
    char* end = nullptr;
    errno = 0;
    const unsigned long long val = std::strtoull(s.c_str(), &end, 10);
    if (errno != 0 || end == s.c_str() || *end != '\0') throw std::runtime_error("Invalid unsigned integer in " + ctx);
    return static_cast<uint64_t>(val);
}

inline std::vector<int> parse_cpu_list(const std::string& raw, const std::string& ctx) {
    std::set<int> cpus;
    std::string norm = raw;
    for (char& c : norm) if (c == ',') c = ' ';
    std::istringstream iss(norm);
    for (std::string tok; iss >> tok;) {
        const size_t dash = tok.find('-');
        if (dash == std::string::npos) {
            cpus.insert(static_cast<int>(parse_u64_strict(tok, ctx)));
        } else {
            const int lo = static_cast<int>(parse_u64_strict(tok.substr(0, dash), ctx));
            const int hi = static_cast<int>(parse_u64_strict(tok.substr(dash + 1U), ctx));
            if (hi < lo || hi - lo > 4096) throw std::runtime_error("Invalid CPU range in " + ctx);
            for (int c = lo; c <= hi; ++c) cpus.insert(c);
        }
    }
    return std::vector<int>(cpus.begin(), cpus.end());
}

inline std::vector<StateSample> parse_time_in_state(const std::string& raw, const std::string& ctx) {
    std::vector<StateSample> states;
    std::set<uint64_t> seen_freqs;
    std::istringstream iss(raw);
    for (std::string line; std::getline(iss, line);) {
        const std::string trimmed = trim_ws(line);
        if (trimmed.empty()) continue;
        std::istringstream ls(trimmed);
        std::string f_tok, t_tok, extra;
        if (!(ls >> f_tok >> t_tok) || (ls >> extra)) {
            throw std::runtime_error("Malformed time_in_state row in " + ctx);
        }
        const uint64_t freq = parse_u64_strict(f_tok, ctx + ":freq");
        const uint64_t ticks = parse_u64_strict(t_tok, ctx + ":ticks");
        if (freq == 0 || !seen_freqs.insert(freq).second) throw std::runtime_error("Invalid frequency in " + ctx);
        states.push_back(StateSample{freq, ticks});
    }
    if (states.empty()) throw std::runtime_error("Empty time_in_state table in " + ctx);
    std::sort(states.begin(), states.end(),
              [](const StateSample& a, const StateSample& b) { return a.freq_khz < b.freq_khz; });
    return states;
}

inline std::string resolve_cpufreq_dir(const std::string& sysroot) {
    for (const auto& rel : {"/sys/devices/system/cpu/cpufreq", "/cpufreq", ""}) {
        const std::string cand = sysroot + rel;
        if (is_directory_path(cand)) return cand;
    }
    throw std::runtime_error("Sysroot directory does not exist: '" + sysroot + "'");
}

inline SystemSnapshot read_sysroot_snapshot(const std::string& sysroot) {
    const std::string cpufreq_dir = resolve_cpufreq_dir(sysroot);
    DIR* dir = ::opendir(cpufreq_dir.c_str());
    if (dir == nullptr) throw std::runtime_error("Cannot open cpufreq directory '" + cpufreq_dir + "'");

    std::vector<std::pair<int, std::string>> policy_dirs;
    while (struct dirent* ent = ::readdir(dir)) {
        const std::string name(ent->d_name);
        if (name.rfind("policy", 0) == 0 && name.size() > 6U) {
            const std::string suffix = name.substr(6U);
            const bool digits = std::all_of(suffix.begin(), suffix.end(),
                                            [](unsigned char c) { return std::isdigit(c) != 0; });
            const std::string full = cpufreq_dir + "/" + name;
            if (digits && is_directory_path(full)) {
                policy_dirs.emplace_back(static_cast<int>(parse_u64_strict(suffix, full)), full);
            }
        }
    }
    (void)::closedir(dir);
    if (policy_dirs.empty()) throw std::runtime_error("No cpufreq policy* directories under '" + cpufreq_dir + "'");
    std::sort(policy_dirs.begin(), policy_dirs.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    struct timespec mono {};
    (void)::clock_gettime(CLOCK_MONOTONIC, &mono);
    SystemSnapshot snap{sysroot, static_cast<int64_t>(mono.tv_sec) * 1000000000LL + mono.tv_nsec, {}};

    for (const auto& [pid, pdir] : policy_dirs) {
        PolicySnapshot p{};
        p.policy_id = pid;
        p.policy_name = "policy" + std::to_string(pid);
        p.related_cpus = parse_cpu_list(read_text_file(pdir + "/related_cpus"), pdir + "/related_cpus");
        const std::string aff_path = pdir + "/affected_cpus";
        p.affected_cpus = is_regular_file_path(aff_path) ? parse_cpu_list(read_text_file(aff_path), aff_path) : p.related_cpus;
        p.scaling_governor = trim_ws(read_text_file(pdir + "/scaling_governor"));
        if (is_regular_file_path(pdir + "/scaling_cur_freq")) {
            p.scaling_cur_freq_khz = parse_u64_strict(read_text_file(pdir + "/scaling_cur_freq"), pdir + "/scaling_cur_freq");
        }
        p.scaling_min_freq_khz = parse_u64_strict(read_text_file(pdir + "/scaling_min_freq"), pdir + "/scaling_min_freq");
        p.scaling_max_freq_khz = parse_u64_strict(read_text_file(pdir + "/scaling_max_freq"), pdir + "/scaling_max_freq");
        p.cpuinfo_min_freq_khz = parse_u64_strict(read_text_file(pdir + "/cpuinfo_min_freq"), pdir + "/cpuinfo_min_freq");
        p.cpuinfo_max_freq_khz = parse_u64_strict(read_text_file(pdir + "/cpuinfo_max_freq"), pdir + "/cpuinfo_max_freq");
        if (p.cpuinfo_max_freq_khz == 0 || p.cpuinfo_min_freq_khz > p.cpuinfo_max_freq_khz) {
            throw std::runtime_error("Invalid cpuinfo frequency bounds in " + pdir);
        }
        if (is_regular_file_path(pdir + "/stats/total_trans")) {
            p.total_trans = parse_u64_strict(read_text_file(pdir + "/stats/total_trans"), pdir + "/stats/total_trans");
            p.has_total_trans = true;
        }
        p.time_in_state = parse_time_in_state(read_text_file(pdir + "/stats/time_in_state"), pdir + "/stats/time_in_state");
        snap.policies.push_back(std::move(p));
    }
    return snap;
}

inline DeltaAuditReport compute_residency_delta(const SystemSnapshot& before,
                                                const SystemSnapshot& after,
                                                double top_fraction = 0.8,
                                                double tick_ms = 10.0) {
    if (!(top_fraction > 0.0 && top_fraction <= 1.0) || !std::isfinite(top_fraction)) {
        throw std::invalid_argument("top_fraction must be in (0, 1]");
    }
    if (!(tick_ms > 0.0) || !std::isfinite(tick_ms)) {
        throw std::invalid_argument("tick_ms must be positive");
    }
    if (before.policies.empty() || before.policies.size() != after.policies.size()) {
        throw std::runtime_error("Policy topology count mismatch between snapshots");
    }

    DeltaAuditReport report{};
    report.top_fraction = top_fraction;
    report.tick_ms = tick_ms;

    std::set<std::string> governors_seen;
    std::vector<double> active_norm_ratios;
    std::vector<double> active_upper_shares;
    double min_cap = 1.0;

    for (size_t idx = 0; idx < before.policies.size(); ++idx) {
        const PolicySnapshot& b = before.policies[idx];
        const PolicySnapshot& a = after.policies[idx];
        if (b.policy_id != a.policy_id || b.related_cpus != a.related_cpus ||
            b.cpuinfo_max_freq_khz != a.cpuinfo_max_freq_khz ||
            b.time_in_state.size() != a.time_in_state.size()) {
            throw std::runtime_error("Policy metadata or state table mismatch on policy" +
                                     std::to_string(b.policy_id));
        }
        if (b.has_total_trans && a.has_total_trans && a.total_trans < b.total_trans) {
            throw std::runtime_error("Counter regression in total_trans on " + b.policy_name);
        }

        PolicyDelta pd{};
        pd.policy_id = b.policy_id;
        pd.policy_name = b.policy_name;
        pd.related_cpus = b.related_cpus;
        pd.before_governor = b.scaling_governor;
        pd.after_governor = a.scaling_governor;
        governors_seen.insert(a.scaling_governor);
        pd.cpuinfo_min_freq_khz = a.cpuinfo_min_freq_khz;
        pd.cpuinfo_max_freq_khz = a.cpuinfo_max_freq_khz;
        pd.before_scaling_max_khz = b.scaling_max_freq_khz;
        pd.after_scaling_max_khz = a.scaling_max_freq_khz;
        pd.before_cap_ratio = b.cap_ratio();
        pd.after_cap_ratio = a.cap_ratio();
        pd.min_cap_ratio = std::min(pd.before_cap_ratio, pd.after_cap_ratio);
        pd.cap_throttled = (b.scaling_max_freq_khz < b.cpuinfo_max_freq_khz) ||
                           (a.scaling_max_freq_khz < a.cpuinfo_max_freq_khz);
        min_cap = std::min(min_cap, pd.min_cap_ratio);
        if (pd.cap_throttled) {
            report.asymmetry.capped_policy_ids.push_back(pd.policy_id);
        }

        pd.upper_band_threshold_khz = static_cast<uint64_t>(
            std::ceil(top_fraction * static_cast<double>(a.cpuinfo_max_freq_khz) - 1e-9));

        long double weighted_khz_sum = 0.0L;
        uint64_t upper_delta_ticks = 0;
        pd.histogram.reserve(b.time_in_state.size());

        for (size_t s = 0; s < b.time_in_state.size(); ++s) {
            const StateSample& bs = b.time_in_state[s];
            const StateSample& as = a.time_in_state[s];
            if (bs.freq_khz != as.freq_khz) {
                throw std::runtime_error("Frequency table mismatch on " + b.policy_name);
            }
            if (as.ticks < bs.ticks) {
                throw std::runtime_error("Counter regression in time_in_state on " + b.policy_name +
                                         " at " + std::to_string(bs.freq_khz) + " kHz");
            }
            const uint64_t dt = as.ticks - bs.ticks;
            const bool upper = bs.freq_khz >= pd.upper_band_threshold_khz;
            pd.total_delta_ticks += dt;
            weighted_khz_sum += static_cast<long double>(bs.freq_khz) * static_cast<long double>(dt);
            if (upper) upper_delta_ticks += dt;
            pd.histogram.push_back(StateDelta{
                bs.freq_khz, bs.ticks, as.ticks, dt, static_cast<double>(dt) * tick_ms, 0.0, upper
            });
        }

        pd.total_delta_ms = static_cast<double>(pd.total_delta_ticks) * tick_ms;
        if (pd.total_delta_ticks > 0) {
            const double denom = static_cast<double>(pd.total_delta_ticks);
            for (auto& bin : pd.histogram) {
                bin.share = static_cast<double>(bin.delta_ticks) / denom;
            }
            pd.weighted_avg_freq_khz =
                static_cast<double>(weighted_khz_sum / static_cast<long double>(pd.total_delta_ticks));
            pd.normalized_freq_ratio =
                pd.weighted_avg_freq_khz / static_cast<double>(pd.cpuinfo_max_freq_khz);
            pd.upper_band_share = static_cast<double>(upper_delta_ticks) / denom;
            active_norm_ratios.push_back(pd.normalized_freq_ratio);
            active_upper_shares.push_back(pd.upper_band_share);
        }

        if (b.has_total_trans && a.has_total_trans) {
            pd.delta_transitions = a.total_trans - b.total_trans;
            const double elapsed_s = pd.total_delta_ms / 1000.0;
            pd.transitions_per_sec =
                elapsed_s > 0.0 ? (static_cast<double>(pd.delta_transitions) / elapsed_s) : 0.0;
        }
        report.policies.push_back(std::move(pd));
    }

    ClusterAsymmetry& asym = report.asymmetry;
    asym.policy_count = report.policies.size();
    asym.active_policy_count = active_norm_ratios.size();
    asym.min_cap_ratio_across_policies = min_cap;
    asym.governor_heterogeneous = governors_seen.size() > 1U;
    if (!active_norm_ratios.empty()) {
        const auto [mn_it, mx_it] = std::minmax_element(active_norm_ratios.begin(), active_norm_ratios.end());
        asym.min_normalized_freq_ratio = *mn_it;
        asym.max_normalized_freq_ratio = *mx_it;
        asym.normalized_freq_spread = *mx_it - *mn_it;

        const auto [umn_it, umx_it] = std::minmax_element(active_upper_shares.begin(), active_upper_shares.end());
        asym.min_upper_band_share = *umn_it;
        asym.max_upper_band_share = *umx_it;
        asym.upper_band_share_spread = *umx_it - *umn_it;
    }
    return report;
}

}  // namespace cpufreq_residency

#endif  // CPUFREQ_RESIDENCY_HPP_
