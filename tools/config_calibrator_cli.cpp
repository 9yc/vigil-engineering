#include "binomial_bounds.hpp"

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
        const ssize_t rc = ::write(fd, payload.data() + offset, payload.size() - offset);
        if (rc < 0) {
            if (errno == EINTR) continue;
            const int saved = errno;
            ::close(fd);
            throw std::runtime_error("write failed for '" + path + "': " + std::strerror(saved));
        }
        offset += static_cast<std::size_t>(rc);
    }
    if (::close(fd) != 0) {
        throw std::runtime_error("close failed for '" + path + "': " + std::strerror(errno));
    }
}

double parse_double_val(const std::string& label, const std::string& raw) {
    std::size_t idx = 0;
    double val = 0.0;
    try {
        val = std::stod(raw, &idx);
    } catch (...) {
        throw std::invalid_argument("invalid float for " + label + ": " + raw);
    }
    if (idx != raw.size() || !std::isfinite(val)) {
        throw std::invalid_argument("invalid float for " + label + ": " + raw);
    }
    return val;
}

std::size_t parse_size_val(const std::string& label, const std::string& raw) {
    const std::string t = trim_str(raw);
    if (t.empty() || t[0] == '-') {
        throw std::invalid_argument("invalid unsigned integer for " + label + ": " + raw);
    }
    std::size_t idx = 0;
    unsigned long long val = 0;
    try {
        val = std::stoull(t, &idx, 10);
    } catch (...) {
        throw std::invalid_argument("invalid unsigned integer for " + label + ": " + raw);
    }
    if (idx != t.size()) throw std::invalid_argument("invalid unsigned integer for " + label + ": " + raw);
    return static_cast<std::size_t>(val);
}

bool parse_bool_val(const std::string& label, const std::string& raw) {
    const std::string t = trim_str(raw);
    if (t == "1" || t == "true" || t == "True" || t == "TRUE") return true;
    if (t == "0" || t == "false" || t == "False" || t == "FALSE") return false;
    throw std::invalid_argument("invalid boolean for " + label + ": " + raw);
}

std::string extract_json_field(const std::string& line, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    const std::size_t kpos = line.find(needle);
    if (kpos == std::string::npos) throw std::invalid_argument("missing JSON field: " + key);
    const std::size_t colon = line.find(':', kpos + needle.size());
    if (colon == std::string::npos) throw std::invalid_argument("malformed JSON field: " + key);
    std::size_t s = colon + 1;
    while (s < line.size() && std::isspace(static_cast<unsigned char>(line[s])) != 0) ++s;
    if (s >= line.size()) throw std::invalid_argument("empty JSON field: " + key);
    if (line[s] == '"') {
        ++s;
        const std::size_t eq = line.find('"', s);
        if (eq == std::string::npos) throw std::invalid_argument("unterminated JSON string: " + key);
        return line.substr(s, eq - s);
    }
    std::size_t e = s;
    while (e < line.size() && line[e] != ',' && line[e] != '}' &&
           std::isspace(static_cast<unsigned char>(line[e])) == 0) {
        ++e;
    }
    return line.substr(s, e - s);
}

std::vector<calib::EvaluationRecord> load_records(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) throw std::runtime_error("unable to open dataset: " + path);
    std::vector<calib::EvaluationRecord> out;
    std::string raw_line;
    std::size_t line_no = 0;
    bool saw_non_comment = false;

    while (std::getline(ifs, raw_line)) {
        ++line_no;
        const std::string line = trim_str(raw_line);
        if (line.empty() || line[0] == '#') continue;
        if (!saw_non_comment && line.rfind("request_id", 0) == 0) {
            saw_non_comment = true;
            continue;
        }
        saw_non_comment = true;

        calib::EvaluationRecord rec;
        if (line.front() == '{') {
            rec.request_id = extract_json_field(line, "request_id");
            rec.config_id = extract_json_field(line, "config_id");
            rec.passed = parse_bool_val("passed", extract_json_field(line, "passed"));
            rec.latency_s = parse_double_val("latency_s", extract_json_field(line, "latency_s"));
            rec.energy_j = parse_double_val("energy_j", extract_json_field(line, "energy_j"));
            rec.tokens = parse_size_val("tokens", extract_json_field(line, "tokens"));
        } else {
            std::vector<std::string> cols;
            std::stringstream ss(line);
            std::string cell;
            while (std::getline(ss, cell, ',')) cols.push_back(trim_str(cell));
            if (cols.size() != 6) {
                throw std::invalid_argument("expected 6 CSV columns at line " + std::to_string(line_no));
            }
            rec.request_id = cols[0];
            rec.config_id = cols[1];
            rec.passed = parse_bool_val("passed", cols[2]);
            rec.latency_s = parse_double_val("latency_s", cols[3]);
            rec.energy_j = parse_double_val("energy_j", cols[4]);
            rec.tokens = parse_size_val("tokens", cols[5]);
        }
        out.push_back(rec);
    }
    return out;
}

std::string run_math_batch(const std::string& batch_path) {
    std::ifstream ifs(batch_path);
    if (!ifs.is_open()) throw std::runtime_error("unable to open math batch file: " + batch_path);
    std::ostringstream oss;
    oss << std::setprecision(17);
    std::string raw_line;
    while (std::getline(ifs, raw_line)) {
        const std::string line = trim_str(raw_line);
        if (line.empty()) continue;
        const std::string op = extract_json_field(line, "op");
        if (op == "betainc") {
            const double a = parse_double_val("a", extract_json_field(line, "a"));
            const double b = parse_double_val("b", extract_json_field(line, "b"));
            const double x = parse_double_val("x", extract_json_field(line, "x"));
            oss << "{\"op\": \"betainc\", \"value\": " << calib::betainc(a, b, x) << "}\n";
        } else if (op == "beta_inv") {
            const double p = parse_double_val("p", extract_json_field(line, "p"));
            const double a = parse_double_val("a", extract_json_field(line, "a"));
            const double b = parse_double_val("b", extract_json_field(line, "b"));
            oss << "{\"op\": \"beta_inv\", \"value\": " << calib::beta_inv(p, a, b) << "}\n";
        } else if (op == "cp") {
            const std::size_t k = parse_size_val("k", extract_json_field(line, "k"));
            const std::size_t n = parse_size_val("n", extract_json_field(line, "n"));
            const double alpha = parse_double_val("alpha", extract_json_field(line, "alpha"));
            const double lo = calib::clopper_pearson_lower(k, n, alpha);
            const double hi = calib::clopper_pearson_upper(k, n, alpha);
            const double tail_at_lo = (k > 0) ? calib::exact_binomial_upper_tail(k, n, lo) : 1.0;
            oss << "{\"op\": \"cp\", \"lower\": " << lo << ", \"upper\": " << hi
                << ", \"upper_tail_at_lower\": " << tail_at_lo << "}\n";
        } else {
            throw std::invalid_argument("unknown math batch op: " + op);
        }
    }
    return oss.str();
}

std::string serialize_report(
    const calib::CalibrationReport& rep, const calib::CalibrationParams& params) {
    std::ostringstream oss;
    oss << std::setprecision(15) << "{\n"
        << "  \"utility_mode\": \""
        << (params.utility_mode == calib::UtilityMode::HardDeadline ? "hard_deadline" : "soft_exp") << "\",\n"
        << "  \"screening_mode\": \""
        << (params.screening_mode == calib::ScreeningMode::ClopperPearsonBonferroni ? "clopper_pearson" : "point") << "\",\n"
        << "  \"target_utility\": " << params.target_utility << ",\n"
        << "  \"delta\": " << params.delta << ",\n"
        << "  \"num_candidates\": " << rep.num_candidates << ",\n"
        << "  \"bonferroni_alpha\": " << rep.bonferroni_alpha << ",\n"
        << "  \"feasible\": " << (rep.feasible ? "true" : "false") << ",\n"
        << "  \"selected_config_id\": "
        << (rep.feasible ? ("\"" + json_escape(rep.selected_config_id) + "\"") : "null") << ",\n"
        << "  \"selected_cal_energy_j\": " << rep.selected_cal_energy_j << ",\n"
        << "  \"selected_cal_utility\": " << rep.selected_cal_utility << ",\n"
        << "  \"selected_cal_certificate\": " << rep.selected_cal_certificate << ",\n"
        << "  \"has_eval\": " << (rep.has_eval ? "true" : "false") << ",\n"
        << "  \"held_out_target_attained\": " << (rep.held_out_target_attained ? "true" : "false") << ",\n"
        << "  \"held_out_utility\": " << rep.held_out_utility << ",\n"
        << "  \"held_out_mean_energy_j\": " << rep.held_out_mean_energy_j << ",\n"
        << "  \"oracle_feasible\": " << (rep.oracle_feasible ? "true" : "false") << ",\n"
        << "  \"oracle_config_id\": "
        << (rep.oracle_feasible ? ("\"" + json_escape(rep.oracle_config_id) + "\"") : "null") << ",\n"
        << "  \"oracle_mean_energy_j\": " << rep.oracle_mean_energy_j << ",\n"
        << "  \"oracle_utility\": " << rep.oracle_utility << ",\n"
        << "  \"oracle_energy_gap_j\": " << rep.oracle_energy_gap_j << ",\n"
        << "  \"oracle_energy_ratio\": " << rep.oracle_energy_ratio << ",\n"
        << "  \"candidates\": [\n";
    for (std::size_t i = 0; i < rep.candidates.size(); ++i) {
        const auto& c = rep.candidates[i];
        oss << "    {\n"
            << "      \"config_id\": \"" << json_escape(c.config_id) << "\",\n"
            << "      \"sample_count\": " << c.sample_count << ",\n"
            << "      \"effective_successes\": " << c.effective_successes << ",\n"
            << "      \"empirical_utility\": " << c.empirical_utility << ",\n"
            << "      \"cp_lower_bound\": " << c.cp_lower_bound << ",\n"
            << "      \"cp_upper_bound\": " << c.cp_upper_bound << ",\n"
            << "      \"screening_score\": " << c.screening_score << ",\n"
            << "      \"cal_feasible\": " << (c.cal_feasible ? "true" : "false") << ",\n"
            << "      \"mean_energy_j\": " << c.mean_energy_j << ",\n"
            << "      \"mean_latency_s\": " << c.mean_latency_s << ",\n"
            << "      \"energy_per_token_j\": " << c.energy_per_token_j << ",\n"
            << "      \"has_eval\": " << (c.has_eval ? "true" : "false") << ",\n"
            << "      \"eval_sample_count\": " << c.eval_sample_count << ",\n"
            << "      \"eval_utility\": " << c.eval_utility << ",\n"
            << "      \"eval_cp_lower_bound\": " << c.eval_cp_lower_bound << ",\n"
            << "      \"eval_mean_energy_j\": " << c.eval_mean_energy_j << ",\n"
            << "      \"eval_mean_latency_s\": " << c.eval_mean_latency_s << ",\n"
            << "      \"eval_attains_target\": " << (c.eval_attains_target ? "true" : "false") << "\n"
            << "    }" << (i + 1 < rep.candidates.size() ? "," : "") << "\n";
    }
    oss << "  ]\n}\n";
    return oss.str();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string cal_path;
        std::string eval_path;
        std::string math_batch_path;
        std::string output_path;
        calib::CalibrationParams params;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto next_val = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument("missing value after flag " + arg);
                return argv[++i];
            };
            if (arg == "--cal") cal_path = next_val();
            else if (arg == "--eval") eval_path = next_val();
            else if (arg == "--math-batch") math_batch_path = next_val();
            else if (arg == "--output") output_path = next_val();
            else if (arg == "--utility-mode") {
                const std::string m = next_val();
                if (m == "hard_deadline") params.utility_mode = calib::UtilityMode::HardDeadline;
                else if (m == "soft_exp") params.utility_mode = calib::UtilityMode::SoftExponential;
                else throw std::invalid_argument("unknown --utility-mode: " + m);
            } else if (arg == "--screening") {
                const std::string s = next_val();
                if (s == "clopper_pearson") params.screening_mode = calib::ScreeningMode::ClopperPearsonBonferroni;
                else if (s == "point") params.screening_mode = calib::ScreeningMode::PointEstimate;
                else throw std::invalid_argument("unknown --screening: " + s);
            } else if (arg == "--deadline-s") params.deadline_s = parse_double_val(arg, next_val());
            else if (arg == "--tau-s") params.tau_s = parse_double_val(arg, next_val());
            else if (arg == "--target") params.target_utility = parse_double_val(arg, next_val());
            else if (arg == "--delta") params.delta = parse_double_val(arg, next_val());
            else throw std::invalid_argument("unknown CLI option: " + arg);
        }

        if (output_path.empty()) throw std::invalid_argument("--output is required");
        if (!math_batch_path.empty()) {
            if (!cal_path.empty() || !eval_path.empty()) {
                throw std::invalid_argument("--math-batch cannot be combined with --cal or --eval");
            }
            write_exclusive_file(output_path, run_math_batch(math_batch_path));
            return 0;
        }
        if (cal_path.empty()) throw std::invalid_argument("--cal is required");

        params.validate();
        const auto cal_records = load_records(cal_path);
        std::vector<calib::EvaluationRecord> eval_records;
        if (!eval_path.empty()) {
            eval_records = load_records(eval_path);
            if (eval_records.empty()) throw std::invalid_argument("held-out evaluation file contains zero records");
        }
        const auto report = calib::calibrate_and_evaluate(cal_records, eval_records, params);
        write_exclusive_file(output_path, serialize_report(report, params));
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "config_calibrator_cli error: " << ex.what() << "\n";
        return 1;
    }
}
