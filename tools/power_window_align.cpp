#include "power_window.hpp"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string json_escape(const std::string& s) {
    std::ostringstream oss;
    for (unsigned char c : s) {
        switch (c) {
            case '"': oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default:
                if (c < 0x20U) {
                    oss << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<unsigned int>(c) << std::dec;
                } else {
                    oss << static_cast<char>(c);
                }
        }
    }
    return oss.str();
}

std::vector<std::string> read_nonempty_lines(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) throw std::runtime_error("Cannot open input file: '" + path + "'");
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(ifs, line)) {
        const std::string t = power_window::trim_str(line);
        if (!t.empty() && t.front() != '#') lines.push_back(t);
    }
    return lines;
}

size_t require_col(const std::map<std::string, size_t>& cols, const std::string& name, const std::string& ctx) {
    auto it = cols.find(name);
    if (it == cols.end()) throw std::runtime_error("Missing required column '" + name + "' in " + ctx);
    return it->second;
}

std::vector<power_window::PowerSample> load_telemetry_stream(const std::string& path) {
    using namespace power_window;
    const std::vector<std::string> lines = read_nonempty_lines(path);
    if (lines.empty()) throw std::runtime_error("Telemetry file is empty: '" + path + "'");

    std::vector<PowerSample> samples;
    if (lines.front().front() == '{') {
        samples.reserve(lines.size());
        for (const auto& line : lines) {
            const auto kv = parse_flat_json_object(line);
            PowerSample s{};
            s.timestamp_ns = parse_i64_field(kv.at("timestamp_ns"), "timestamp_ns");
            s.voltage_v = parse_double_field(kv.at("voltage_v"), "voltage_v");
            s.current_a = parse_double_field(kv.at("current_a"), "current_a");
            s.valid = kv.count("valid") ? parse_bool_field(kv.at("valid"), "valid") : true;
            samples.push_back(s);
        }
    } else {
        const std::vector<std::string> hdr = split_csv_line(lines.front());
        std::map<std::string, size_t> col_idx;
        for (size_t i = 0; i < hdr.size(); ++i) col_idx[hdr[i]] = i;
        const size_t c_ts = require_col(col_idx, "timestamp_ns", path);
        const size_t c_v = require_col(col_idx, "voltage_v", path);
        const size_t c_i = require_col(col_idx, "current_a", path);
        const bool has_valid = col_idx.count("valid") > 0;
        const size_t c_val = has_valid ? col_idx["valid"] : 0;

        samples.reserve(lines.size() - 1U);
        for (size_t r = 1U; r < lines.size(); ++r) {
            const std::vector<std::string> row = split_csv_line(lines[r]);
            if (row.size() != hdr.size()) {
                throw std::runtime_error("CSV column count mismatch in " + path + " at row " + std::to_string(r));
            }
            PowerSample s{};
            s.timestamp_ns = parse_i64_field(row[c_ts], "timestamp_ns");
            s.voltage_v = parse_double_field(row[c_v], "voltage_v");
            s.current_a = parse_double_field(row[c_i], "current_a");
            s.valid = has_valid ? parse_bool_field(row[c_val], "valid") : true;
            samples.push_back(s);
        }
    }
    validate_sorted_telemetry(samples);
    return samples;
}

std::vector<power_window::WorkloadWindow> load_workload_windows(const std::string& path) {
    using namespace power_window;
    const std::vector<std::string> lines = read_nonempty_lines(path);
    if (lines.empty()) throw std::runtime_error("Workload windows file is empty: '" + path + "'");

    std::vector<WorkloadWindow> windows;
    if (lines.front().front() == '{') {
        windows.reserve(lines.size());
        for (const auto& line : lines) {
            const auto kv = parse_flat_json_object(line);
            WorkloadWindow w{};
            w.request_id = kv.at("request_id");
            w.start_ns = parse_i64_field(kv.at("start_ns"), "start_ns");
            w.end_ns = parse_i64_field(kv.at("end_ns"), "end_ns");
            w.output_tokens = kv.count("output_tokens")
                                  ? parse_u64_field(kv.at("output_tokens"), "output_tokens")
                                  : 0ULL;
            windows.push_back(std::move(w));
        }
    } else {
        const std::vector<std::string> hdr = split_csv_line(lines.front());
        std::map<std::string, size_t> col_idx;
        for (size_t i = 0; i < hdr.size(); ++i) col_idx[hdr[i]] = i;
        const size_t c_id = require_col(col_idx, "request_id", path);
        const size_t c_st = require_col(col_idx, "start_ns", path);
        const size_t c_en = require_col(col_idx, "end_ns", path);
        const bool has_tok = col_idx.count("output_tokens") > 0;
        const size_t c_tok = has_tok ? col_idx["output_tokens"] : 0;

        windows.reserve(lines.size() - 1U);
        for (size_t r = 1U; r < lines.size(); ++r) {
            const std::vector<std::string> row = split_csv_line(lines[r]);
            if (row.size() != hdr.size()) {
                throw std::runtime_error("Window CSV column mismatch at row " + std::to_string(r));
            }
            WorkloadWindow w{};
            w.request_id = row[c_id];
            w.start_ns = parse_i64_field(row[c_st], "start_ns");
            w.end_ns = parse_i64_field(row[c_en], "end_ns");
            w.output_tokens = has_tok ? parse_u64_field(row[c_tok], "output_tokens") : 0ULL;
            windows.push_back(std::move(w));
        }
    }
    if (windows.empty()) throw std::runtime_error("Zero workload windows parsed from '" + path + "'");
    return windows;
}

std::string format_alignment_report_json(const power_window::IntegrationConfig& cfg,
                                         const std::vector<power_window::AlignedWindowResult>& results,
                                         const power_window::AggregateSummary& agg) {
    using namespace power_window;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(9);
    oss << "{\n"
        << "  \"config\": {\n"
        << "    \"max_gap_ns\": " << cfg.max_gap_ns << ",\n"
        << "    \"idle_lead_ns\": " << cfg.idle_lead_ns << ",\n"
        << "    \"idle_trail_ns\": " << cfg.idle_trail_ns << ",\n"
        << "    \"idle_guard_ns\": " << cfg.idle_guard_ns << ",\n"
        << "    \"has_fixed_idle_w\": " << (cfg.has_fixed_idle_w ? "true" : "false") << ",\n"
        << "    \"fixed_idle_w\": " << cfg.fixed_idle_w << "\n"
        << "  },\n"
        << "  \"aggregate\": {\n"
        << "    \"total_windows\": " << agg.total_windows << ",\n"
        << "    \"accepted_windows\": " << agg.accepted_windows << ",\n"
        << "    \"rejected_windows\": " << agg.rejected_windows << ",\n"
        << "    \"total_output_tokens\": " << agg.total_output_tokens << ",\n"
        << "    \"total_duration_s\": " << agg.total_duration_s << ",\n"
        << "    \"total_gross_trapezoid_j\": " << agg.total_gross_trapezoid_j << ",\n"
        << "    \"total_gross_midpoint_j\": " << agg.total_gross_midpoint_j << ",\n"
        << "    \"total_net_trapezoid_j\": " << agg.total_net_trapezoid_j << ",\n"
        << "    \"total_net_midpoint_j\": " << agg.total_net_midpoint_j << ",\n"
        << "    \"mean_gross_trapezoid_w\": " << agg.mean_gross_trapezoid_w << ",\n"
        << "    \"mean_net_trapezoid_w\": " << agg.mean_net_trapezoid_w << ",\n"
        << "    \"aggregate_gross_trapezoid_j_per_token\": " << agg.aggregate_gross_trapezoid_j_per_token << ",\n"
        << "    \"aggregate_net_trapezoid_j_per_token\": " << agg.aggregate_net_trapezoid_j_per_token << ",\n"
        << "    \"aggregate_net_midpoint_j_per_token\": " << agg.aggregate_net_midpoint_j_per_token << "\n"
        << "  },\n"
        << "  \"windows\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        oss << "    {\n"
            << "      \"request_id\": \"" << json_escape(r.request_id) << "\",\n"
            << "      \"start_ns\": " << r.start_ns << ",\n"
            << "      \"end_ns\": " << r.end_ns << ",\n"
            << "      \"output_tokens\": " << r.output_tokens << ",\n"
            << "      \"status\": \"" << reject_code_name(r.status) << "\",\n"
            << "      \"reject_detail\": \"" << json_escape(r.reject_detail) << "\",\n"
            << "      \"duration_s\": " << r.duration_s << ",\n"
            << "      \"max_observed_gap_ns\": " << r.max_observed_gap_ns << ",\n"
            << "      \"spanned_samples\": " << r.spanned_samples << ",\n"
            << "      \"gross_trapezoid_j\": " << r.gross_trapezoid_j << ",\n"
            << "      \"gross_midpoint_j\": " << r.gross_midpoint_j << ",\n"
            << "      \"mean_gross_trapezoid_w\": " << r.mean_gross_trapezoid_w << ",\n"
            << "      \"mean_gross_midpoint_w\": " << r.mean_gross_midpoint_w << ",\n"
            << "      \"baseline_applied\": " << (r.baseline_applied ? "true" : "false") << ",\n"
            << "      \"idle_baseline_trapezoid_w\": " << r.idle_baseline_trapezoid_w << ",\n"
            << "      \"idle_baseline_midpoint_w\": " << r.idle_baseline_midpoint_w << ",\n"
            << "      \"idle_energy_trapezoid_j\": " << r.idle_energy_trapezoid_j << ",\n"
            << "      \"idle_energy_midpoint_j\": " << r.idle_energy_midpoint_j << ",\n"
            << "      \"net_trapezoid_j\": " << r.net_trapezoid_j << ",\n"
            << "      \"net_midpoint_j\": " << r.net_midpoint_j << ",\n"
            << "      \"gross_trapezoid_j_per_token\": " << r.gross_trapezoid_j_per_token << ",\n"
            << "      \"gross_midpoint_j_per_token\": " << r.gross_midpoint_j_per_token << ",\n"
            << "      \"net_trapezoid_j_per_token\": " << r.net_trapezoid_j_per_token << ",\n"
            << "      \"net_midpoint_j_per_token\": " << r.net_midpoint_j_per_token << "\n"
            << "    }" << (i + 1U < results.size() ? "," : "") << "\n";
    }
    oss << "  ]\n}\n";
    return oss.str();
}

void print_usage(std::ostream& out, const char* prog) {
    out << "Usage: " << prog << " --telemetry <csv_or_jsonl> --windows <csv_or_jsonl> --output <json> [options]\n"
        << "  --max-gap-ms <num>           Maximum permissible sample gap in ms (default: 50)\n"
        << "  --idle-lead-ms <num>         Pre-window idle baseline duration in ms (default: 0)\n"
        << "  --idle-trail-ms <num>        Post-window idle baseline duration in ms (default: 0)\n"
        << "  --idle-guard-ms <num>        Guard margin separating idle and active spans in ms (default: 0)\n"
        << "  --fixed-idle-w <num>         Explicit constant idle baseline power in Watts\n"
        << "  --allow-rejected-windows     Emit JSON report even when one or more windows fail checks\n";
}

}  // namespace

int main(int argc, char** argv) {
    using namespace power_window;
    try {
        std::string telemetry_path;
        std::string windows_path;
        std::string output_path;
        IntegrationConfig cfg{};
        bool allow_rejected = false;

        auto next_arg = [&](int& idx, const std::string& flag) -> std::string {
            if (idx + 1 >= argc) throw std::invalid_argument("Missing value for " + flag);
            return std::string(argv[++idx]);
        };
        auto parse_nonneg_ms = [&](const std::string& raw, const std::string& flag) -> int64_t {
            const double ms = parse_double_field(raw, flag);
            if (ms < 0.0) throw std::invalid_argument(flag + " must be non-negative");
            return static_cast<int64_t>(std::llround(ms * static_cast<double>(kNsPerMs)));
        };

        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (arg == "--help" || arg == "-h") {
                print_usage(std::cout, argv[0]);
                return 0;
            }
            if (arg == "--telemetry") telemetry_path = next_arg(i, arg);
            else if (arg == "--windows") windows_path = next_arg(i, arg);
            else if (arg == "--output") output_path = next_arg(i, arg);
            else if (arg == "--max-gap-ms") {
                const double ms = parse_double_field(next_arg(i, arg), arg);
                if (ms <= 0.0) throw std::invalid_argument("--max-gap-ms must be positive");
                cfg.max_gap_ns = std::max<int64_t>(1, std::llround(ms * static_cast<double>(kNsPerMs)));
            } else if (arg == "--idle-lead-ms") {
                cfg.idle_lead_ns = parse_nonneg_ms(next_arg(i, arg), arg);
            } else if (arg == "--idle-trail-ms") {
                cfg.idle_trail_ns = parse_nonneg_ms(next_arg(i, arg), arg);
            } else if (arg == "--idle-guard-ms") {
                cfg.idle_guard_ns = parse_nonneg_ms(next_arg(i, arg), arg);
            } else if (arg == "--fixed-idle-w") {
                cfg.fixed_idle_w = parse_double_field(next_arg(i, arg), arg);
                if (cfg.fixed_idle_w < 0.0) throw std::invalid_argument("--fixed-idle-w must be >= 0");
                cfg.has_fixed_idle_w = true;
            } else if (arg == "--allow-rejected-windows") {
                allow_rejected = true;
            } else {
                throw std::invalid_argument("Unknown option: " + arg);
            }
        }

        if (telemetry_path.empty() || windows_path.empty() || output_path.empty()) {
            throw std::invalid_argument("--telemetry, --windows, and --output are required");
        }
        if (::access(output_path.c_str(), F_OK) == 0) {
            throw std::runtime_error("exclusive open rejected for '" + output_path + "': File exists");
        }

        const std::vector<PowerSample> samples = load_telemetry_stream(telemetry_path);
        const std::vector<WorkloadWindow> windows = load_workload_windows(windows_path);

        std::vector<AlignedWindowResult> results;
        results.reserve(windows.size());
        for (const auto& w : windows) {
            const AlignedWindowResult r = evaluate_workload_window(samples, w, cfg);
            if (r.status != RejectCode::kOk && !allow_rejected) {
                throw std::runtime_error("Window '" + w.request_id + "' rejected (" +
                                         std::string(reject_code_name(r.status)) + "): " + r.reject_detail);
            }
            results.push_back(r);
        }

        const AggregateSummary agg = summarize_aligned_windows(results);
        write_exclusive_output(output_path, format_alignment_report_json(cfg, results, agg));
        return agg.rejected_windows == 0 ? 0 : 1;
    } catch (const std::exception& ex) {
        std::cerr << "power_window_align error: " << ex.what() << "\n";
        return 1;
    }
}
