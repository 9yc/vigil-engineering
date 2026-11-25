#include "cpufreq_residency.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>
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

std::string format_int_array(const std::vector<int>& vals) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < vals.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << vals[i];
    }
    oss << "]";
    return oss.str();
}

std::string serialize_snapshot_json(const cpufreq_residency::SystemSnapshot& snap) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6) << "{\n"
        << "  \"kind\": \"cpufreq_snapshot\", \"source_label\": \"" << json_escape(snap.source_label)
        << "\", \"captured_monotonic_ns\": " << snap.captured_monotonic_ns << ",\n  \"policies\": [\n";
    for (size_t i = 0; i < snap.policies.size(); ++i) {
        const auto& p = snap.policies[i];
        oss << "    {\"policy_id\": " << p.policy_id
            << ", \"policy_name\": \"" << json_escape(p.policy_name)
            << "\", \"related_cpus\": " << format_int_array(p.related_cpus)
            << ", \"affected_cpus\": " << format_int_array(p.affected_cpus)
            << ", \"scaling_governor\": \"" << json_escape(p.scaling_governor)
            << "\", \"scaling_cur_freq_khz\": " << p.scaling_cur_freq_khz
            << ", \"scaling_min_freq_khz\": " << p.scaling_min_freq_khz
            << ", \"scaling_max_freq_khz\": " << p.scaling_max_freq_khz
            << ", \"cpuinfo_min_freq_khz\": " << p.cpuinfo_min_freq_khz
            << ", \"cpuinfo_max_freq_khz\": " << p.cpuinfo_max_freq_khz
            << ", \"cap_ratio\": " << p.cap_ratio()
            << ", \"cap_throttled\": " << (p.scaling_max_freq_khz < p.cpuinfo_max_freq_khz ? "true" : "false")
            << ", \"has_total_trans\": " << (p.has_total_trans ? "true" : "false")
            << ", \"total_trans\": " << p.total_trans
            << ", \"total_ticks\": " << p.total_ticks()
            << ", \"weighted_mean_freq_khz\": " << p.weighted_mean_freq_khz()
            << ", \"time_in_state\": [";
        for (size_t j = 0; j < p.time_in_state.size(); ++j) {
            const auto& st = p.time_in_state[j];
            if (j > 0) oss << ", ";
            oss << "{\"freq_khz\": " << st.freq_khz << ", \"ticks\": " << st.ticks << "}";
        }
        oss << "]}" << (i + 1U < snap.policies.size() ? "," : "") << "\n";
    }
    oss << "  ]\n}\n";
    return oss.str();
}

std::string serialize_delta_json(const cpufreq_residency::DeltaAuditReport& rep) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6) << "{\n"
        << "  \"kind\": \"cpufreq_delta_audit\", \"top_fraction\": " << rep.top_fraction
        << ", \"tick_ms\": " << rep.tick_ms << ",\n  \"cluster_asymmetry\": {"
        << "\"policy_count\": " << rep.asymmetry.policy_count
        << ", \"active_policy_count\": " << rep.asymmetry.active_policy_count
        << ", \"max_normalized_freq_ratio\": " << rep.asymmetry.max_normalized_freq_ratio
        << ", \"min_normalized_freq_ratio\": " << rep.asymmetry.min_normalized_freq_ratio
        << ", \"normalized_freq_spread\": " << rep.asymmetry.normalized_freq_spread
        << ", \"max_upper_band_share\": " << rep.asymmetry.max_upper_band_share
        << ", \"min_upper_band_share\": " << rep.asymmetry.min_upper_band_share
        << ", \"upper_band_share_spread\": " << rep.asymmetry.upper_band_share_spread
        << ", \"min_cap_ratio_across_policies\": " << rep.asymmetry.min_cap_ratio_across_policies
        << ", \"governor_heterogeneous\": " << (rep.asymmetry.governor_heterogeneous ? "true" : "false")
        << ", \"capped_policy_ids\": " << format_int_array(rep.asymmetry.capped_policy_ids)
        << "},\n  \"policies\": [\n";
    for (size_t i = 0; i < rep.policies.size(); ++i) {
        const auto& p = rep.policies[i];
        oss << "    {\"policy_id\": " << p.policy_id
            << ", \"policy_name\": \"" << json_escape(p.policy_name)
            << "\", \"related_cpus\": " << format_int_array(p.related_cpus)
            << ", \"before_governor\": \"" << json_escape(p.before_governor)
            << "\", \"after_governor\": \"" << json_escape(p.after_governor)
            << "\", \"cpuinfo_min_freq_khz\": " << p.cpuinfo_min_freq_khz
            << ", \"cpuinfo_max_freq_khz\": " << p.cpuinfo_max_freq_khz
            << ", \"before_scaling_max_khz\": " << p.before_scaling_max_khz
            << ", \"after_scaling_max_khz\": " << p.after_scaling_max_khz
            << ", \"before_cap_ratio\": " << p.before_cap_ratio
            << ", \"after_cap_ratio\": " << p.after_cap_ratio
            << ", \"min_cap_ratio\": " << p.min_cap_ratio
            << ", \"cap_throttled\": " << (p.cap_throttled ? "true" : "false")
            << ", \"total_delta_ticks\": " << p.total_delta_ticks
            << ", \"total_delta_ms\": " << p.total_delta_ms
            << ", \"weighted_avg_freq_khz\": " << p.weighted_avg_freq_khz
            << ", \"normalized_freq_ratio\": " << p.normalized_freq_ratio
            << ", \"upper_band_threshold_khz\": " << p.upper_band_threshold_khz
            << ", \"upper_band_share\": " << p.upper_band_share
            << ", \"delta_transitions\": " << p.delta_transitions
            << ", \"transitions_per_sec\": " << p.transitions_per_sec
            << ", \"histogram\": [";
        for (size_t j = 0; j < p.histogram.size(); ++j) {
            const auto& h = p.histogram[j];
            if (j > 0) oss << ", ";
            oss << "{\"freq_khz\": " << h.freq_khz
                << ", \"before_ticks\": " << h.before_ticks
                << ", \"after_ticks\": " << h.after_ticks
                << ", \"delta_ticks\": " << h.delta_ticks
                << ", \"delta_ms\": " << h.delta_ms
                << ", \"share\": " << h.share
                << ", \"in_upper_band\": " << (h.in_upper_band ? "true" : "false") << "}";
        }
        oss << "]}" << (i + 1U < rep.policies.size() ? "," : "") << "\n";
    }
    oss << "  ]\n}\n";
    return oss.str();
}

struct JsonValue {
    using Object = std::map<std::string, JsonValue>;
    using Array = std::vector<JsonValue>;
    std::variant<std::nullptr_t, bool, double, std::string, Array, Object> data;

    const Object& as_object() const { return std::get<Object>(data); }
    const Array& as_array() const { return std::get<Array>(data); }
    const std::string& as_string() const { return std::get<std::string>(data); }
    bool as_bool() const { return std::get<bool>(data); }
    uint64_t as_u64() const {
        const double d = std::get<double>(data);
        if (d < 0.0 || !std::isfinite(d)) throw std::runtime_error("Negative JSON integer");
        return static_cast<uint64_t>(std::llround(d));
    }
};

class MiniJsonParser {
public:
    explicit MiniJsonParser(const std::string& text) : s_(text) {}

    JsonValue parse() {
        skip_ws();
        JsonValue v = parse_value();
        skip_ws();
        if (pos_ != s_.size()) throw std::runtime_error("Trailing characters in JSON snapshot");
        return v;
    }

private:
    void skip_ws() { while (pos_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[pos_]))) ++pos_; }
    char peek() const {
        if (pos_ >= s_.size()) throw std::runtime_error("Unexpected EOF in JSON");
        return s_[pos_];
    }

    JsonValue parse_value() {
        skip_ws();
        const char c = peek();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return JsonValue{parse_string()};
        if (c == 't' && s_.compare(pos_, 4, "true") == 0) { pos_ += 4; return JsonValue{true}; }
        if (c == 'f' && s_.compare(pos_, 5, "false") == 0) { pos_ += 5; return JsonValue{false}; }
        if (c == 'n' && s_.compare(pos_, 4, "null") == 0) { pos_ += 4; return JsonValue{nullptr}; }
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return parse_number();
        throw std::runtime_error("Unexpected token in JSON");
    }

    std::string parse_string() {
        ++pos_;
        std::string out;
        while (pos_ < s_.size()) {
            const char c = s_[pos_++];
            if (c == '"') return out;
            if (c == '\\') {
                if (pos_ >= s_.size()) throw std::runtime_error("Unterminated JSON escape");
                const char esc = s_[pos_++];
                if (esc == 'n') out.push_back('\n');
                else if (esc == 'r') out.push_back('\r');
                else if (esc == 't') out.push_back('\t');
                else out.push_back(esc);
            } else {
                out.push_back(c);
            }
        }
        throw std::runtime_error("Unterminated JSON string");
    }

    JsonValue parse_number() {
        const char* start = s_.c_str() + pos_;
        char* end = nullptr;
        errno = 0;
        const double val = std::strtod(start, &end);
        if (errno != 0 || end == start) throw std::runtime_error("Invalid JSON number");
        pos_ = static_cast<size_t>(end - s_.c_str());
        return JsonValue{val};
    }

    JsonValue parse_array() {
        ++pos_;
        JsonValue::Array arr;
        skip_ws();
        if (peek() == ']') { ++pos_; return JsonValue{std::move(arr)}; }
        while (true) {
            arr.push_back(parse_value());
            skip_ws();
            const char sep = s_[pos_++];
            if (sep == ']') break;
            if (sep != ',') throw std::runtime_error("Expected ',' or ']' in JSON array");
        }
        return JsonValue{std::move(arr)};
    }

    JsonValue parse_object() {
        ++pos_;
        JsonValue::Object obj;
        skip_ws();
        if (peek() == '}') { ++pos_; return JsonValue{std::move(obj)}; }
        while (true) {
            skip_ws();
            if (peek() != '"') throw std::runtime_error("Expected string key in JSON object");
            std::string key = parse_string();
            skip_ws();
            if (peek() != ':') throw std::runtime_error("Expected ':' in JSON object");
            ++pos_;
            obj.emplace(std::move(key), parse_value());
            skip_ws();
            const char sep = s_[pos_++];
            if (sep == '}') break;
            if (sep != ',') throw std::runtime_error("Expected ',' or '}' in JSON object");
        }
        return JsonValue{std::move(obj)};
    }

    const std::string& s_;
    size_t pos_ = 0;
};

cpufreq_residency::SystemSnapshot load_snapshot_from_json_file(const std::string& path) {
    using namespace cpufreq_residency;
    const JsonValue root = MiniJsonParser(read_text_file(path)).parse();
    const auto& obj = root.as_object();
    SystemSnapshot snap{};
    snap.source_label = path;
    if (auto it = obj.find("captured_monotonic_ns"); it != obj.end()) {
        snap.captured_monotonic_ns = static_cast<int64_t>(it->second.as_u64());
    }
    for (const auto& pv : obj.at("policies").as_array()) {
        const auto& po = pv.as_object();
        PolicySnapshot p{};
        p.policy_id = static_cast<int>(po.at("policy_id").as_u64());
        p.policy_name = po.at("policy_name").as_string();
        for (const auto& c : po.at("related_cpus").as_array()) p.related_cpus.push_back(static_cast<int>(c.as_u64()));
        for (const auto& c : po.at("affected_cpus").as_array()) p.affected_cpus.push_back(static_cast<int>(c.as_u64()));
        p.scaling_governor = po.at("scaling_governor").as_string();
        p.scaling_cur_freq_khz = po.at("scaling_cur_freq_khz").as_u64();
        p.scaling_min_freq_khz = po.at("scaling_min_freq_khz").as_u64();
        p.scaling_max_freq_khz = po.at("scaling_max_freq_khz").as_u64();
        p.cpuinfo_min_freq_khz = po.at("cpuinfo_min_freq_khz").as_u64();
        p.cpuinfo_max_freq_khz = po.at("cpuinfo_max_freq_khz").as_u64();
        p.has_total_trans = po.at("has_total_trans").as_bool();
        p.total_trans = po.at("total_trans").as_u64();
        for (const auto& sv : po.at("time_in_state").as_array()) {
            const auto& so = sv.as_object();
            p.time_in_state.push_back(StateSample{so.at("freq_khz").as_u64(), so.at("ticks").as_u64()});
        }
        snap.policies.push_back(std::move(p));
    }
    return snap;
}

cpufreq_residency::SystemSnapshot load_snapshot_any(const std::string& path) {
    if (cpufreq_residency::is_directory_path(path)) return cpufreq_residency::read_sysroot_snapshot(path);
    if (cpufreq_residency::is_regular_file_path(path)) return load_snapshot_from_json_file(path);
    throw std::runtime_error("Snapshot source not found: '" + path + "'");
}

void print_usage(std::ostream& out, const char* prog) {
    out << "Usage:\n  " << prog << " snapshot --sysroot <dir> --output <json>\n  "
        << prog << " delta --before <dir_or_json> --after <dir_or_json> [--top-fraction 0.8] [--tick-ms 10] --output <json>\n";
}

}  // namespace

int main(int argc, char** argv) {
    using namespace cpufreq_residency;
    try {
        std::string mode, sysroot, before_src, after_src, output_path;
        double top_fraction = 0.8, tick_ms = 10.0;
        auto next_arg = [&](int& idx, const std::string& flag) -> std::string {
            if (idx + 1 >= argc) throw std::invalid_argument("Missing argument for " + flag);
            return std::string(argv[++idx]);
        };
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (arg == "--help" || arg == "-h") { print_usage(std::cout, argv[0]); return 0; }
            if ((arg == "snapshot" || arg == "delta") && mode.empty()) mode = arg;
            else if (arg == "--mode") mode = next_arg(i, arg);
            else if (arg == "--sysroot") sysroot = next_arg(i, arg);
            else if (arg == "--before") before_src = next_arg(i, arg);
            else if (arg == "--after") after_src = next_arg(i, arg);
            else if (arg == "--output") output_path = next_arg(i, arg);
            else if (arg == "--top-fraction") {
                const std::string raw = next_arg(i, arg);
                char* end = nullptr;
                top_fraction = std::strtod(raw.c_str(), &end);
                if (end == raw.c_str() || *end != '\0') throw std::invalid_argument("Invalid --top-fraction");
            } else if (arg == "--tick-ms") {
                const std::string raw = next_arg(i, arg);
                char* end = nullptr;
                tick_ms = std::strtod(raw.c_str(), &end);
                if (end == raw.c_str() || *end != '\0') throw std::invalid_argument("Invalid --tick-ms");
            } else {
                throw std::invalid_argument("Unrecognized argument: " + arg);
            }
        }

        if (output_path.empty()) throw std::invalid_argument("--output <path> is required");
        if (mode == "snapshot") {
            if (sysroot.empty()) throw std::invalid_argument("snapshot mode requires --sysroot <dir>");
            const SystemSnapshot snap = read_sysroot_snapshot(sysroot);
            write_exclusive_file(output_path, serialize_snapshot_json(snap));
            return 0;
        }
        if (mode == "delta") {
            if (before_src.empty() || after_src.empty()) {
                throw std::invalid_argument("delta mode requires --before and --after");
            }
            const SystemSnapshot before = load_snapshot_any(before_src);
            const SystemSnapshot after = load_snapshot_any(after_src);
            const DeltaAuditReport report = compute_residency_delta(before, after, top_fraction, tick_ms);
            write_exclusive_file(output_path, serialize_delta_json(report));
            return 0;
        }
        throw std::invalid_argument("Mode must be 'snapshot' or 'delta'");
    } catch (const std::exception& ex) {
        std::cerr << "dvfs_residency_audit error: " << ex.what() << "\n";
        return 1;
    }
}
