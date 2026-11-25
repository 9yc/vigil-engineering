#include "thermal_cooldown.hpp"

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

std::string trim_s(const std::string& s) {
    size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1U]))) --e;
    return s.substr(b, e - b);
}

std::string json_escape(const std::string& s) {
    std::ostringstream oss;
    for (unsigned char c : s) {
        switch (c) {
            case '"': oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default: oss << static_cast<char>(c); break;
        }
    }
    return oss.str();
}

double parse_num(const std::string& raw, const std::string& name) {
    const std::string s = trim_s(raw);
    char* end = nullptr;
    errno = 0;
    const double v = std::strtod(s.c_str(), &end);
    if (errno != 0 || end == s.c_str() || *end != '\0' || !std::isfinite(v)) {
        throw std::invalid_argument("Invalid numeric value for " + name + ": '" + s + "'");
    }
    return v;
}

std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> out;
    std::istringstream iss(line);
    std::string cell;
    while (std::getline(iss, cell, ',')) out.push_back(trim_s(cell));
    if (!line.empty() && line.back() == ',') out.emplace_back("");
    return out;
}

std::map<std::string, std::string> parse_jsonl_flat(const std::string& raw) {
    const std::string s = trim_s(raw);
    if (s.size() < 2U || s.front() != '{' || s.back() != '}') throw std::runtime_error("Malformed JSONL: " + s);
    std::map<std::string, std::string> kv;
    size_t pos = 1U, end_pos = s.size() - 1U;
    auto skip_ws = [&]() { while (pos < end_pos && std::isspace(static_cast<unsigned char>(s[pos]))) ++pos; };
    auto read_str = [&]() -> std::string {
        if (pos >= end_pos || s[pos] != '"') throw std::runtime_error("Expected JSON string");
        ++pos;
        std::string res;
        while (pos < end_pos) {
            const char c = s[pos++];
            if (c == '"') return res;
            res.push_back((c == '\\' && pos < end_pos) ? s[pos++] : c);
        }
        throw std::runtime_error("Unterminated JSON string");
    };
    for (skip_ws(); pos < end_pos;) {
        const std::string k = read_str();
        skip_ws();
        if (pos >= end_pos || s[pos++] != ':') throw std::runtime_error("Expected ':' in JSONL");
        skip_ws();
        std::string v;
        if (pos < end_pos && s[pos] == '"') {
            v = read_str();
        } else {
            const size_t st = pos;
            while (pos < end_pos && s[pos] != ',') ++pos;
            v = trim_s(s.substr(st, pos - st));
        }
        kv[k] = v;
        skip_ws();
        if (pos < end_pos) {
            if (s[pos++] != ',') throw std::runtime_error("Expected ',' in JSONL");
            skip_ws();
        }
    }
    return kv;
}

std::vector<std::string> read_lines_clean(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) throw std::runtime_error("Unable to open input '" + path + "'");
    std::vector<std::string> lines;
    for (std::string line; std::getline(ifs, line);) {
        const std::string t = trim_s(line);
        if (!t.empty() && t.front() != '#') lines.push_back(t);
    }
    if (lines.empty()) throw std::runtime_error("Empty input file '" + path + "'");
    return lines;
}

std::vector<thermal_cooldown::ThermalSample> load_trace(const std::string& path) {
    const auto lines = read_lines_clean(path);
    std::vector<thermal_cooldown::ThermalSample> out;
    if (lines.front().front() == '{') {
        for (const auto& l : lines) {
            const auto kv = parse_jsonl_flat(l);
            out.push_back({parse_num(kv.at("time_s"), "time_s"), parse_num(kv.at("temp_c"), "temp_c")});
        }
        return out;
    }
    const auto hdr = split_csv(lines.front());
    std::map<std::string, size_t> cols;
    for (size_t i = 0; i < hdr.size(); ++i) cols[hdr[i]] = i;
    if (!cols.count("time_s") || !cols.count("temp_c")) throw std::runtime_error("Trace CSV requires time_s,temp_c");
    for (size_t r = 1U; r < lines.size(); ++r) {
        const auto row = split_csv(lines[r]);
        if (row.size() != hdr.size()) throw std::runtime_error("Malformed trace CSV row");
        out.push_back({parse_num(row[cols["time_s"]], "time_s"), parse_num(row[cols["temp_c"]], "temp_c")});
    }
    return out;
}

std::vector<thermal_cooldown::BatchThermalRecord> load_batches(const std::string& path) {
    const auto lines = read_lines_clean(path);
    std::vector<thermal_cooldown::BatchThermalRecord> out;
    if (lines.front().front() == '{') {
        for (const auto& l : lines) {
            const auto kv = parse_jsonl_flat(l);
            thermal_cooldown::BatchThermalRecord b{kv.at("batch_id"), 0.0, 0.0,
                parse_num(kv.at("start_temp_c"), "start_temp_c"), parse_num(kv.at("end_temp_c"), "end_temp_c")};
            if (kv.count("start_time_s")) b.start_time_s = parse_num(kv.at("start_time_s"), "start_time_s");
            if (kv.count("end_time_s")) b.end_time_s = parse_num(kv.at("end_time_s"), "end_time_s");
            out.push_back(std::move(b));
        }
        return out;
    }
    const auto hdr = split_csv(lines.front());
    std::map<std::string, size_t> cols;
    for (size_t i = 0; i < hdr.size(); ++i) cols[hdr[i]] = i;
    if (cols.count("batch_id") && cols.count("start_temp_c") && cols.count("end_temp_c")) {
        for (size_t r = 1U; r < lines.size(); ++r) {
            const auto row = split_csv(lines[r]);
            if (row.size() != hdr.size()) throw std::runtime_error("Malformed batch CSV row");
            thermal_cooldown::BatchThermalRecord b{row[cols["batch_id"]], 0.0, 0.0,
                parse_num(row[cols["start_temp_c"]], "start_temp_c"), parse_num(row[cols["end_temp_c"]], "end_temp_c")};
            if (cols.count("start_time_s")) b.start_time_s = parse_num(row[cols["start_time_s"]], "start_time_s");
            if (cols.count("end_time_s")) b.end_time_s = parse_num(row[cols["end_time_s"]], "end_time_s");
            out.push_back(std::move(b));
        }
        return out;
    }
    if (cols.count("batch_id") && cols.count("time_s") && cols.count("temp_c")) {
        std::map<std::string, size_t> idx_by_id;
        for (size_t r = 1U; r < lines.size(); ++r) {
            const auto row = split_csv(lines[r]);
            if (row.size() != hdr.size()) throw std::runtime_error("Malformed grouped batch CSV row");
            const std::string& bid = row[cols["batch_id"]];
            const double ts = parse_num(row[cols["time_s"]], "time_s"), tc = parse_num(row[cols["temp_c"]], "temp_c");
            if (auto it = idx_by_id.find(bid); it == idx_by_id.end()) {
                idx_by_id[bid] = out.size();
                out.push_back({bid, ts, ts, tc, tc});
            } else {
                auto& rec = out[it->second];
                if (ts < rec.start_time_s) { rec.start_time_s = ts; rec.start_temp_c = tc; }
                if (ts >= rec.end_time_s) { rec.end_time_s = ts; rec.end_temp_c = tc; }
            }
        }
        return out;
    }
    throw std::runtime_error("Batch CSV requires batch_id with (start_temp_c,end_temp_c) or (time_s,temp_c)");
}

void print_usage(std::ostream& out, const char* prog) {
    out << "Usage: " << prog << " [--trace <csv_or_jsonl>] [--batches <csv_or_jsonl>] --output <json> [options]\n";
}

}  // namespace

int main(int argc, char** argv) {
    using namespace thermal_cooldown;
    try {
        std::string trace_path, batches_path, output_path;
        bool has_target = false, fix_ambient = false, fail_on_creep = false;
        double target_c = 0.0, ambient_c = 0.0;
        PlateauConfig plateau_cfg{};
        BatchDriftThresholds drift_limits{};
        auto next_arg = [&](int& idx, const std::string& flag) -> std::string {
            if (idx + 1 >= argc) throw std::invalid_argument("Missing value for " + flag);
            return std::string(argv[++idx]);
        };
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (arg == "--help" || arg == "-h") { print_usage(std::cout, argv[0]); return 0; }
            if (arg == "--trace") trace_path = next_arg(i, arg);
            else if (arg == "--batches") batches_path = next_arg(i, arg);
            else if (arg == "--output") output_path = next_arg(i, arg);
            else if (arg == "--target-c") { target_c = parse_num(next_arg(i, arg), arg); has_target = true; }
            else if (arg == "--ambient-c") { ambient_c = parse_num(next_arg(i, arg), arg); fix_ambient = true; }
            else if (arg == "--hold-s") plateau_cfg.hold_s = parse_num(next_arg(i, arg), arg);
            else if (arg == "--max-slope-c-per-s") plateau_cfg.max_slope_c_per_s = parse_num(next_arg(i, arg), arg);
            else if (arg == "--max-span-c") plateau_cfg.max_span_c = parse_num(next_arg(i, arg), arg);
            else if (arg == "--max-total-drift-c") drift_limits.max_total_start_drift_c = parse_num(next_arg(i, arg), arg);
            else if (arg == "--max-drift-per-batch-c") drift_limits.max_slope_c_per_batch = parse_num(next_arg(i, arg), arg);
            else if (arg == "--max-step-rise-c") drift_limits.max_step_start_rise_c = parse_num(next_arg(i, arg), arg);
            else if (arg == "--fail-on-creep") fail_on_creep = true;
            else throw std::invalid_argument("Unrecognized option: " + arg);
        }

        if (output_path.empty()) throw std::invalid_argument("--output <path> is required");
        if (trace_path.empty() && batches_path.empty()) {
            throw std::invalid_argument("At least one of --trace or --batches must be specified");
        }
        if (::access(output_path.c_str(), F_OK) == 0) {
            throw std::runtime_error("exclusive open rejected for '" + output_path + "': File exists");
        }

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6) << "{\n";
        bool need_comma = false;
        bool creep_flagged = false;

        if (!trace_path.empty()) {
            const auto trace = load_trace(trace_path);
            const CoolingFitResult fit = fit_newton_cooling(trace, fix_ambient, ambient_c);
            const PlateauReport plat = detect_steady_state_plateau(trace, plateau_cfg);
            TargetCooldownEstimate te{};
            if (has_target) te = estimate_cooldown_to_target(trace, fit, target_c);

            oss << "  \"newton_cooling\": {\"converged\": " << (fit.converged ? "true" : "false")
                << ", \"ref_time_s\": " << fit.ref_time_s << ", \"t_amb_c\": " << fit.t_amb_c
                << ", \"t_initial_c\": " << fit.t_initial_c << ", \"tau_s\": " << fit.tau_s
                << ", \"rmse_c\": " << fit.rmse_c << ", \"max_abs_error_c\": " << fit.max_abs_error_c
                << ", \"r_squared\": " << fit.r_squared << ", \"iterations\": " << fit.iterations << "},\n"
                << "  \"target_cooldown\": {\"requested\": " << (te.requested ? "true" : "false")
                << ", \"target_temp_c\": " << te.target_temp_c
                << ", \"already_at_or_below_target\": " << (te.already_at_or_below_target ? "true" : "false")
                << ", \"reachable\": " << (te.reachable ? "true" : "false")
                << ", \"predicted_elapsed_from_start_s\": " << te.predicted_elapsed_from_start_s
                << ", \"predicted_target_time_s\": " << te.predicted_target_time_s
                << ", \"predicted_remaining_from_end_s\": " << te.predicted_remaining_from_end_s
                << ", \"empirical_crossed\": " << (te.empirical_crossed ? "true" : "false")
                << ", \"empirical_crossing_time_s\": " << te.empirical_crossing_time_s << "},\n"
                << "  \"plateau\": {\"detected\": " << (plat.detected ? "true" : "false")
                << ", \"settled_at_end\": " << (plat.settled_at_end ? "true" : "false")
                << ", \"onset_time_s\": " << plat.onset_time_s
                << ", \"window_end_time_s\": " << plat.window_end_time_s
                << ", \"mean_temp_c\": " << plat.mean_temp_c
                << ", \"slope_c_per_s\": " << plat.slope_c_per_s
                << ", \"span_c\": " << plat.span_c
                << ", \"trailing_mean_temp_c\": " << plat.trailing_mean_temp_c
                << ", \"trailing_slope_c_per_s\": " << plat.trailing_slope_c_per_s
                << ", \"trailing_span_c\": " << plat.trailing_span_c << "}";
            need_comma = true;
        }

        if (!batches_path.empty()) {
            const auto batches = load_batches(batches_path);
            const BatchDriftReport d = audit_batch_thermal_drift(batches, drift_limits);
            creep_flagged = d.thermal_creep_detected;
            if (need_comma) oss << ",\n";
            oss << "  \"batch_drift\": {\"batch_count\": " << d.batch_count
                << ", \"initial_start_temp_c\": " << d.initial_start_temp_c
                << ", \"final_start_temp_c\": " << d.final_start_temp_c
                << ", \"total_start_drift_c\": " << d.total_start_drift_c
                << ", \"total_end_drift_c\": " << d.total_end_drift_c
                << ", \"start_slope_c_per_batch\": " << d.start_slope_c_per_batch
                << ", \"end_slope_c_per_batch\": " << d.end_slope_c_per_batch
                << ", \"max_step_start_rise_c\": " << d.max_step_start_rise_c
                << ", \"mean_intra_batch_rise_c\": " << d.mean_intra_batch_rise_c
                << ", \"thermal_creep_detected\": " << (d.thermal_creep_detected ? "true" : "false")
                << ", \"creep_reasons\": [";
            for (size_t i = 0; i < d.creep_reasons.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << "\"" << json_escape(d.creep_reasons[i]) << "\"";
            }
            oss << "]}";
        }
        oss << "\n}\n";

        write_exclusive_text(output_path, oss.str());
        return (fail_on_creep && creep_flagged) ? 1 : 0;
    } catch (const std::exception& ex) {
        std::cerr << "cooldown_detector error: " << ex.what() << "\n";
        return 2;
    }
}
