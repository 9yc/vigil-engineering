#include "cadence_controller.hpp"

#include <cctype>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

std::string trim_str(const std::string& s) {
    std::size_t a = 0;
    while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a])) != 0) ++a;
    if (a == s.size()) return "";
    std::size_t b = s.size() - 1;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b])) != 0) --b;
    return s.substr(a, b - a + 1);
}

std::string json_escape(const std::string& s) {
    std::ostringstream oss;
    for (char c : s) {
        if (c == '"') oss << "\\\"";
        else if (c == '\\') oss << "\\\\";
        else if (c == '\n') oss << "\\n";
        else if (c == '\r') oss << "\\r";
        else if (c == '\t') oss << "\\t";
        else oss << c;
    }
    return oss.str();
}

void write_exclusive_file(const std::string& path, const std::string& payload) {
    if (path.empty()) throw std::invalid_argument("output file path must not be empty");
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (fd < 0) {
        throw std::runtime_error("exclusive open failed for '" + path + "': " + std::strerror(errno));
    }
    std::size_t offset = 0;
    while (offset < payload.size()) {
        const ssize_t n = ::write(fd, payload.data() + offset, payload.size() - offset);
        if (n < 0) {
            if (errno == EINTR) continue;
            const int saved_err = errno;
            ::close(fd);
            throw std::runtime_error("write failed for '" + path + "': " + std::strerror(saved_err));
        }
        offset += static_cast<std::size_t>(n);
    }
    if (::close(fd) != 0) {
        throw std::runtime_error("close failed for '" + path + "': " + std::strerror(errno));
    }
}

double parse_double_arg(const std::string& flag, const std::string& raw) {
    std::size_t idx = 0;
    double val = 0.0;
    try {
        val = std::stod(raw, &idx);
    } catch (...) {
        throw std::invalid_argument("invalid numeric value for " + flag + ": " + raw);
    }
    if (idx != raw.size() || !std::isfinite(val)) {
        throw std::invalid_argument("invalid numeric value for " + flag + ": " + raw);
    }
    return val;
}

std::string extract_json_field(const std::string& line, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    const std::size_t kpos = line.find(needle);
    if (kpos == std::string::npos) throw std::invalid_argument("missing JSON field: " + key);
    const std::size_t colon = line.find(':', kpos + needle.size());
    if (colon == std::string::npos) throw std::invalid_argument("malformed JSON field: " + key);
    std::size_t start = colon + 1;
    while (start < line.size() && std::isspace(static_cast<unsigned char>(line[start])) != 0) ++start;
    if (start >= line.size()) throw std::invalid_argument("empty JSON value for: " + key);
    if (line[start] == '"') {
        ++start;
        const std::size_t end_q = line.find('"', start);
        if (end_q == std::string::npos) throw std::invalid_argument("unterminated JSON string: " + key);
        return line.substr(start, end_q - start);
    }
    std::size_t end = start;
    while (end < line.size() && line[end] != ',' && line[end] != '}' &&
           std::isspace(static_cast<unsigned char>(line[end])) == 0) {
        ++end;
    }
    return line.substr(start, end - start);
}

std::vector<cadence::RequestSpec> load_trace(const std::string& input_path) {
    std::ifstream ifs(input_path);
    if (!ifs.is_open()) throw std::runtime_error("unable to open trace file: " + input_path);

    std::vector<cadence::RequestSpec> trace;
    std::string raw_line;
    std::size_t line_no = 0;
    bool saw_data = false;

    while (std::getline(ifs, raw_line)) {
        ++line_no;
        const std::string line = trim_str(raw_line);
        if (line.empty() || line[0] == '#') continue;
        if (!saw_data && line.rfind("request_id", 0) == 0) {
            saw_data = true;
            continue;
        }
        saw_data = true;

        cadence::RequestSpec req;
        if (line.front() == '{') {
            req.request_id = extract_json_field(line, "request_id");
            req.arrival_s = parse_double_arg("arrival_s", extract_json_field(line, "arrival_s"));
            req.compute_units = parse_double_arg("compute_units", extract_json_field(line, "compute_units"));
            req.deadline_s = parse_double_arg("deadline_s", extract_json_field(line, "deadline_s"));
        } else {
            std::vector<std::string> cols;
            std::stringstream ss(line);
            std::string cell;
            while (std::getline(ss, cell, ',')) cols.push_back(trim_str(cell));
            if (cols.size() != 4) {
                throw std::invalid_argument("expected 4 CSV columns at line " + std::to_string(line_no));
            }
            req.request_id = cols[0];
            req.arrival_s = parse_double_arg("arrival_s", cols[1]);
            req.compute_units = parse_double_arg("compute_units", cols[2]);
            req.deadline_s = parse_double_arg("deadline_s", cols[3]);
        }
        trace.push_back(req);
    }
    if (trace.empty()) throw std::invalid_argument("trace file contains zero requests");
    return trace;
}

std::vector<cadence::ExecutionMode> parse_mode_list(const std::string& spec) {
    std::vector<cadence::ExecutionMode> modes;
    std::stringstream ss(spec);
    std::string item;
    while (std::getline(ss, item, ',')) {
        const std::string cleaned = trim_str(item);
        if (!cleaned.empty()) modes.push_back(cadence::parse_execution_mode(cleaned));
    }
    if (modes.empty()) throw std::invalid_argument("mode list must not be empty");
    return modes;
}

void emit_summary_json(
    std::ostringstream& oss, const cadence::SimulationSummary& s, bool include_outcomes) {
    oss << "    {\n"
        << "      \"mode\": \"" << json_escape(s.mode_name) << "\",\n"
        << "      \"total_requests\": " << s.total_requests << ",\n"
        << "      \"timely_requests\": " << s.timely_requests << ",\n"
        << "      \"timely_completion_rate\": " << s.timely_completion_rate << ",\n"
        << "      \"p50_latency_s\": " << s.p50_latency_s << ",\n"
        << "      \"p95_latency_s\": " << s.p95_latency_s << ",\n"
        << "      \"p99_latency_s\": " << s.p99_latency_s << ",\n"
        << "      \"mean_latency_s\": " << s.mean_latency_s << ",\n"
        << "      \"max_latency_s\": " << s.max_latency_s << ",\n"
        << "      \"total_duration_s\": " << s.total_duration_s << ",\n"
        << "      \"active_time_s\": " << s.active_time_s << ",\n"
        << "      \"active_duty_factor\": " << s.active_duty_factor << ",\n"
        << "      \"total_energy_j\": " << s.total_energy_j << ",\n"
        << "      \"active_energy_j\": " << s.active_energy_j << ",\n"
        << "      \"idle_energy_j\": " << s.idle_energy_j << ",\n"
        << "      \"peak_temp_c\": " << s.peak_temp_c << ",\n"
        << "      \"final_temp_c\": " << s.final_temp_c << ",\n"
        << "      \"throttled_duration_s\": " << s.throttled_duration_s;
    if (include_outcomes) {
        oss << ",\n      \"outcomes\": [\n";
        for (std::size_t i = 0; i < s.outcomes.size(); ++i) {
            const auto& o = s.outcomes[i];
            oss << "        {\"request_id\": \"" << json_escape(o.request_id)
                << "\", \"arrival_s\": " << o.arrival_s
                << ", \"start_s\": " << o.start_s
                << ", \"finish_s\": " << o.finish_s
                << ", \"latency_s\": " << o.latency_s
                << ", \"deadline_s\": " << o.deadline_s
                << ", \"met_deadline\": " << (o.met_deadline ? "true" : "false")
                << ", \"energy_j\": " << o.energy_j
                << ", \"peak_temp_c\": " << o.peak_temp_c
                << ", \"throttled_s\": " << o.throttled_s << "}"
                << (i + 1 < s.outcomes.size() ? "," : "") << "\n";
        }
        oss << "      ]\n";
    } else {
        oss << "\n";
    }
    oss << "    }";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string input_path;
        std::string output_path;
        std::string modes_spec = "burst,fixed_sleep,token_bucket,thermal_feedback";
        bool include_outcomes = false;

        cadence::ThermalModelConfig thermal_cfg;
        cadence::PolicyConfig base_policy;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto next_val = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument("missing value after flag " + arg);
                return argv[++i];
            };
            if (arg == "--input") input_path = next_val();
            else if (arg == "--output") output_path = next_val();
            else if (arg == "--modes" || arg == "--mode") modes_spec = next_val();
            else if (arg == "--include-outcomes") include_outcomes = true;
            else if (arg == "--ambient-temp-c") thermal_cfg.ambient_temp_c = parse_double_arg(arg, next_val());
            else if (arg == "--initial-temp-c") thermal_cfg.initial_temp_c = parse_double_arg(arg, next_val());
            else if (arg == "--c-th") thermal_cfg.c_th_j_per_c = parse_double_arg(arg, next_val());
            else if (arg == "--r-th") thermal_cfg.r_th_c_per_w = parse_double_arg(arg, next_val());
            else if (arg == "--idle-power-w") thermal_cfg.idle_power_w = parse_double_arg(arg, next_val());
            else if (arg == "--nominal-power-w") thermal_cfg.nominal_power_w = parse_double_arg(arg, next_val());
            else if (arg == "--nominal-rate") thermal_cfg.nominal_rate_cu_per_s = parse_double_arg(arg, next_val());
            else if (arg == "--throttle-temp-c") thermal_cfg.throttle_temp_c = parse_double_arg(arg, next_val());
            else if (arg == "--severe-throttle-temp-c") thermal_cfg.severe_throttle_temp_c = parse_double_arg(arg, next_val());
            else if (arg == "--throttle-scale") thermal_cfg.throttle_rate_scale = parse_double_arg(arg, next_val());
            else if (arg == "--severe-scale") thermal_cfg.severe_rate_scale = parse_double_arg(arg, next_val());
            else if (arg == "--power-exponent") thermal_cfg.power_exponent = parse_double_arg(arg, next_val());
            else if (arg == "--throttle-leakage-w") thermal_cfg.throttle_leakage_w = parse_double_arg(arg, next_val());
            else if (arg == "--dt-s") thermal_cfg.integration_dt_s = parse_double_arg(arg, next_val());
            else if (arg == "--fixed-sleep-s") base_policy.fixed_sleep_s = parse_double_arg(arg, next_val());
            else if (arg == "--bucket-rate") base_policy.bucket_rate_cu_per_s = parse_double_arg(arg, next_val());
            else if (arg == "--bucket-capacity") base_policy.bucket_capacity_cu = parse_double_arg(arg, next_val());
            else if (arg == "--target-temp-c") base_policy.target_temp_c = parse_double_arg(arg, next_val());
            else if (arg == "--hysteresis-c") base_policy.hysteresis_band_c = parse_double_arg(arg, next_val());
            else if (arg == "--kp") base_policy.kp = parse_double_arg(arg, next_val());
            else if (arg == "--ki") base_policy.ki = parse_double_arg(arg, next_val());
            else if (arg == "--min-scale") base_policy.min_rate_scale = parse_double_arg(arg, next_val());
            else if (arg == "--max-scale") base_policy.max_rate_scale = parse_double_arg(arg, next_val());
            else throw std::invalid_argument("unknown CLI option: " + arg);
        }

        if (input_path.empty() || output_path.empty()) {
            throw std::invalid_argument("both --input and --output are required");
        }
        thermal_cfg.validate();
        base_policy.validate();
        const auto modes = parse_mode_list(modes_spec);
        const auto trace = load_trace(input_path);

        std::vector<cadence::SimulationSummary> summaries;
        summaries.reserve(modes.size());
        for (const auto m : modes) {
            cadence::PolicyConfig p = base_policy;
            p.mode = m;
            cadence::CadenceSimulator sim(thermal_cfg, p);
            summaries.push_back(sim.run(trace));
        }

        std::ostringstream oss;
        oss << std::setprecision(12)
            << "{\n"
            << "  \"request_count\": " << trace.size() << ",\n"
            << "  \"thermal_tau_s\": " << (thermal_cfg.r_th_c_per_w * thermal_cfg.c_th_j_per_c) << ",\n"
            << "  \"idle_equilibrium_temp_c\": "
            << cadence::steady_state_temperature(thermal_cfg.idle_power_w, thermal_cfg) << ",\n"
            << "  \"nominal_equilibrium_temp_c\": "
            << cadence::steady_state_temperature(thermal_cfg.nominal_power_w, thermal_cfg) << ",\n"
            << "  \"results\": [\n";
        for (std::size_t i = 0; i < summaries.size(); ++i) {
            emit_summary_json(oss, summaries[i], include_outcomes);
            oss << (i + 1 < summaries.size() ? "," : "") << "\n";
        }
        oss << "  ],\n  \"by_mode\": {\n";
        for (std::size_t i = 0; i < summaries.size(); ++i) {
            const auto& s = summaries[i];
            oss << "    \"" << json_escape(s.mode_name) << "\": {\n"
                << "      \"timely_completion_rate\": " << s.timely_completion_rate << ",\n"
                << "      \"p50_latency_s\": " << s.p50_latency_s << ",\n"
                << "      \"p95_latency_s\": " << s.p95_latency_s << ",\n"
                << "      \"p99_latency_s\": " << s.p99_latency_s << ",\n"
                << "      \"active_duty_factor\": " << s.active_duty_factor << ",\n"
                << "      \"total_energy_j\": " << s.total_energy_j << ",\n"
                << "      \"peak_temp_c\": " << s.peak_temp_c << ",\n"
                << "      \"final_temp_c\": " << s.final_temp_c << ",\n"
                << "      \"throttled_duration_s\": " << s.throttled_duration_s << "\n"
                << "    }" << (i + 1 < summaries.size() ? "," : "") << "\n";
        }
        oss << "  }\n}\n";

        write_exclusive_file(output_path, oss.str());
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "cadence_sim_cli error: " << ex.what() << "\n";
        return 1;
    }
}
