#ifndef FRAMED_TELEMETRY_HPP
#define FRAMED_TELEMETRY_HPP

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace telemetry {

enum class CurrentUnit { MilliAmps, Amps };

struct DecoderConfig {
    bool allow_partial_tail = false;
    CurrentUnit ibat_unit = CurrentUnit::MilliAmps;
    double min_voltage_v = 3.20;
    double max_voltage_v = 4.60;
    double max_abs_current_a = 15.0;
    int required_battery_status = 3;
    int required_ibat_status = 0;

    void validate() const {
        if (!(min_voltage_v > 0.0 && max_voltage_v > min_voltage_v)) {
            throw std::invalid_argument("invalid voltage window in DecoderConfig");
        }
        if (!(max_abs_current_a > 0.0) || !std::isfinite(max_abs_current_a)) {
            throw std::invalid_argument("max_abs_current_a must be positive and finite");
        }
    }
};

struct ThermalReading {
    bool present = false;
    double value = 0.0;
    int type = -1;
    std::string name;
    int status = -1;
};

struct TelemetrySample {
    std::size_t block_index = 0;
    std::int64_t ts_begin_ms = 0;
    std::int64_t ts_end_ms = 0;
    double ts_mid_ms = 0.0;
    double sample_span_ms = 0.0;

    bool ac_powered = false;
    bool usb_powered = false;
    bool wireless_powered = false;
    bool dock_powered = false;
    int battery_status = 0;
    double battery_voltage_mv = 0.0;
    double voltage_v = 0.0;
    int battery_level = -1;
    double battery_temp_c = 0.0;
    bool has_battery_level = false;
    bool has_battery_temp = false;

    double current_a = 0.0;
    int ibat_status = -1;
    double vbat_v = 0.0;
    bool has_vbat = false;
    double skin_temp_c = 0.0;
    bool has_skin_temp = false;
    double power_w = 0.0;

    std::int64_t pid = 0;
    std::int64_t rss_kb = 0;
    std::int64_t vmhwm_kb = 0;
};

struct RejectedBlock {
    std::size_t block_index = 0;
    std::size_t start_line = 0;
    std::string primary_reason;
    std::vector<std::string> all_reasons;
};

struct StreamDecodeResult {
    std::size_t total_blocks_seen = 0;
    std::size_t complete_blocks = 0;
    std::size_t partial_tail_blocks = 0;
    std::size_t valid_count = 0;
    std::size_t rejected_count = 0;
    std::map<std::string, std::size_t> rejected_by_reason;
    std::vector<TelemetrySample> valid_samples;
    std::vector<RejectedBlock> rejected_blocks;

    double mean_voltage_v = 0.0;
    double mean_abs_current_a = 0.0;
    double mean_power_w = 0.0;
    double max_skin_temp_c = 0.0;
    std::int64_t max_rss_kb = 0;
    std::int64_t max_vmhwm_kb = 0;
};

namespace detail {

inline std::string trim(const std::string& s) {
    std::size_t first = 0;
    while (first < s.size() && std::isspace(static_cast<unsigned char>(s[first])) != 0) ++first;
    if (first == s.size()) return "";
    std::size_t last = s.size() - 1;
    while (last > first && std::isspace(static_cast<unsigned char>(s[last])) != 0) --last;
    return s.substr(first, last - first + 1);
}

inline bool parse_int64_strict(const std::string& raw, std::int64_t* out) {
    const std::string t = trim(raw);
    if (t.empty()) return false;
    try {
        std::size_t idx = 0;
        const long long val = std::stoll(t, &idx, 10);
        if (idx != t.size()) return false;
        *out = static_cast<std::int64_t>(val);
        return true;
    } catch (...) {
        return false;
    }
}

inline bool parse_double_strict(const std::string& raw, double* out) {
    const std::string t = trim(raw);
    if (t.empty()) return false;
    try {
        std::size_t idx = 0;
        const double val = std::stod(t, &idx);
        if (idx != t.size() || !std::isfinite(val)) return false;
        *out = val;
        return true;
    } catch (...) {
        return false;
    }
}

inline bool parse_bool_strict(const std::string& raw, bool* out) {
    const std::string t = trim(raw);
    if (t == "true" || t == "TRUE" || t == "True") {
        *out = true;
        return true;
    }
    if (t == "false" || t == "FALSE" || t == "False") {
        *out = false;
        return true;
    }
    return false;
}

inline bool parse_thermal_entry(const std::string& line, ThermalReading* out) {
    const std::string marker = "Temperature{";
    const std::size_t pos = line.find(marker);
    if (pos == std::string::npos) return false;
    const std::size_t close_pos = line.find('}', pos + marker.size());
    if (close_pos == std::string::npos) return false;
    const std::string body = line.substr(pos + marker.size(), close_pos - (pos + marker.size()));

    bool got_val = false, got_type = false, got_name = false, got_status = false;
    ThermalReading reading;

    std::size_t cursor = 0;
    while (cursor < body.size()) {
        const std::size_t comma = body.find(',', cursor);
        const std::string token = trim(
            comma == std::string::npos ? body.substr(cursor) : body.substr(cursor, comma - cursor));
        const std::size_t eq = token.find('=');
        if (eq != std::string::npos) {
            const std::string k = trim(token.substr(0, eq));
            std::string v = trim(token.substr(eq + 1));
            if (v.size() >= 2 &&
                ((v.front() == '"' && v.back() == '"') || (v.front() == '\'' && v.back() == '\''))) {
                v = v.substr(1, v.size() - 2);
            }
            if (k == "mValue") {
                got_val = parse_double_strict(v, &reading.value);
            } else if (k == "mType") {
                std::int64_t tmp = 0;
                if (parse_int64_strict(v, &tmp)) {
                    reading.type = static_cast<int>(tmp);
                    got_type = true;
                }
            } else if (k == "mName" && !v.empty()) {
                reading.name = v;
                got_name = true;
            } else if (k == "mStatus") {
                std::int64_t tmp = 0;
                if (parse_int64_strict(v, &tmp)) {
                    reading.status = static_cast<int>(tmp);
                    got_status = true;
                }
            }
        }
        if (comma == std::string::npos) break;
        cursor = comma + 1;
    }

    if (got_val && got_type && got_name && got_status) {
        reading.present = true;
        *out = reading;
        return true;
    }
    return false;
}

inline void parse_kv_tokens_from_line(
    const std::string& line,
    bool* has_pid, std::int64_t* pid,
    bool* has_rss, std::int64_t* rss_kb,
    bool* has_vmhwm, std::int64_t* vmhwm_kb) {
    std::string normalized = line;
    for (char& ch : normalized) {
        if (ch == ',' || ch == ';') ch = ' ';
    }
    std::istringstream iss(normalized);
    std::string tok;
    while (iss >> tok) {
        const std::size_t eq = tok.find('=');
        if (eq == std::string::npos) continue;
        const std::string k = tok.substr(0, eq);
        const std::string v = tok.substr(eq + 1);
        std::int64_t parsed = 0;
        if (k == "pid" && parse_int64_strict(v, &parsed)) {
            *pid = parsed;
            *has_pid = true;
        } else if (k == "rss_kb" && parse_int64_strict(v, &parsed)) {
            *rss_kb = parsed;
            *has_rss = true;
        } else if (k == "vmhwm_kb" && parse_int64_strict(v, &parsed)) {
            *vmhwm_kb = parsed;
            *has_vmhwm = true;
        }
    }
}

}  // namespace detail

class TelemetryBlockDecoder {
public:
    explicit TelemetryBlockDecoder(DecoderConfig cfg = DecoderConfig{}) : cfg_(cfg) {
        cfg_.validate();
    }

    StreamDecodeResult decode_stream(const std::string& raw_stream) const {
        StreamDecodeResult result;
        std::istringstream input(raw_stream);
        std::string raw_line;
        std::size_t line_no = 0;

        bool in_block = false;
        std::size_t current_block_idx = 0;
        std::size_t block_start_line = 0;
        std::vector<std::string> block_lines;

        auto record_rejection = [&](std::size_t blk_idx, std::size_t start_ln,
                                    const std::vector<std::string>& reasons) {
            result.rejected_blocks.push_back({blk_idx, start_ln, reasons.front(), reasons});
            ++result.rejected_count;
            ++result.rejected_by_reason[reasons.front()];
        };

        while (std::getline(input, raw_line)) {
            ++line_no;
            if (!raw_line.empty() && raw_line.back() == '\r') raw_line.pop_back();
            const std::string trimmed = detail::trim(raw_line);

            if (trimmed == "__SAMPLE_BEGIN__") {
                if (in_block) {
                    record_rejection(current_block_idx, block_start_line, {"unclosed_interior_block"});
                    block_lines.clear();
                }
                in_block = true;
                current_block_idx = result.total_blocks_seen++;
                block_start_line = line_no;
                block_lines.clear();
                continue;
            }
            if (trimmed == "__SAMPLE_END__") {
                if (!in_block) {
                    record_rejection(result.total_blocks_seen++, line_no, {"orphan_sample_end"});
                    continue;
                }
                in_block = false;
                ++result.complete_blocks;
                evaluate_complete_block(
                    current_block_idx, block_start_line, block_lines, &result, record_rejection);
                block_lines.clear();
                continue;
            }
            if (in_block) block_lines.push_back(trimmed);
        }

        if (in_block) {
            ++result.partial_tail_blocks;
            if (!cfg_.allow_partial_tail) {
                throw std::runtime_error(
                    "truncated telemetry block at stream tail starting at line " +
                    std::to_string(block_start_line));
            }
            record_rejection(current_block_idx, block_start_line, {"truncated_tail"});
        }

        if (!result.valid_samples.empty()) {
            double sum_v = 0.0, sum_i = 0.0, sum_p = 0.0;
            double max_skin = -std::numeric_limits<double>::infinity();
            bool any_skin = false;
            std::int64_t max_rss = 0, max_hwm = 0;

            for (const auto& s : result.valid_samples) {
                sum_v += s.voltage_v;
                sum_i += std::abs(s.current_a);
                sum_p += s.power_w;
                if (s.has_skin_temp) {
                    max_skin = std::max(max_skin, s.skin_temp_c);
                    any_skin = true;
                }
                max_rss = std::max(max_rss, s.rss_kb);
                max_hwm = std::max(max_hwm, s.vmhwm_kb);
            }
            const double denom = static_cast<double>(result.valid_samples.size());
            result.mean_voltage_v = sum_v / denom;
            result.mean_abs_current_a = sum_i / denom;
            result.mean_power_w = sum_p / denom;
            result.max_skin_temp_c = any_skin ? max_skin : 0.0;
            result.max_rss_kb = max_rss;
            result.max_vmhwm_kb = max_hwm;
        }
        return result;
    }

private:
    DecoderConfig cfg_;

    template <typename RejectFn>
    void evaluate_complete_block(
        std::size_t block_idx, std::size_t start_line, const std::vector<std::string>& lines,
        StreamDecodeResult* result, RejectFn&& record_rejection) const {
        TelemetrySample sample;
        sample.block_index = block_idx;

        bool has_ts_begin = false, has_ts_end = false;
        bool has_ac = false, has_usb = false, has_wireless = false, has_dock = false;
        bool has_bat_status = false, has_bat_voltage = false;
        bool has_pid = false, has_rss = false, has_vmhwm = false;
        ThermalReading ibat, vbat, skin;

        for (const std::string& line : lines) {
            if (line.empty()) continue;
            if (line.rfind("__TS_BEGIN__=", 0) == 0) {
                has_ts_begin = detail::parse_int64_strict(line.substr(13), &sample.ts_begin_ms);
                continue;
            }
            if (line.rfind("__TS_END__=", 0) == 0) {
                has_ts_end = detail::parse_int64_strict(line.substr(11), &sample.ts_end_ms);
                continue;
            }
            ThermalReading tr;
            if (detail::parse_thermal_entry(line, &tr)) {
                if (tr.name == "ibat") ibat = tr;
                else if (tr.name == "vbat") vbat = tr;
                else if (tr.name == "skin") skin = tr;
                continue;
            }
            const std::size_t colon = line.find(':');
            if (colon != std::string::npos && line.find('=') == std::string::npos) {
                const std::string key = detail::trim(line.substr(0, colon));
                const std::string val = detail::trim(line.substr(colon + 1));
                if (key == "AC powered") has_ac = detail::parse_bool_strict(val, &sample.ac_powered);
                else if (key == "USB powered") has_usb = detail::parse_bool_strict(val, &sample.usb_powered);
                else if (key == "Wireless powered") has_wireless = detail::parse_bool_strict(val, &sample.wireless_powered);
                else if (key == "Dock powered") has_dock = detail::parse_bool_strict(val, &sample.dock_powered);
                else if (key == "status") {
                    std::int64_t st = 0;
                    if (detail::parse_int64_strict(val, &st)) {
                        sample.battery_status = static_cast<int>(st);
                        has_bat_status = true;
                    }
                } else if (key == "voltage") {
                    if (detail::parse_double_strict(val, &sample.battery_voltage_mv)) {
                        sample.voltage_v = sample.battery_voltage_mv / 1000.0;
                        has_bat_voltage = true;
                    }
                } else if (key == "level") {
                    std::int64_t lv = 0;
                    if (detail::parse_int64_strict(val, &lv)) {
                        sample.battery_level = static_cast<int>(lv);
                        sample.has_battery_level = true;
                    }
                } else if (key == "temperature") {
                    double tenths = 0.0;
                    if (detail::parse_double_strict(val, &tenths)) {
                        sample.battery_temp_c = tenths / 10.0;
                        sample.has_battery_temp = true;
                    }
                }
                continue;
            }
            detail::parse_kv_tokens_from_line(
                line, &has_pid, &sample.pid, &has_rss, &sample.rss_kb, &has_vmhwm, &sample.vmhwm_kb);
        }

        std::vector<std::string> reasons;
        if (!(has_ts_begin && has_ts_end && has_ac && has_usb && has_wireless && has_dock &&
              has_bat_status && has_bat_voltage && ibat.present && has_pid && has_rss && has_vmhwm)) {
            reasons.push_back("missing_required_fields");
        }
        if (has_ts_begin && has_ts_end) {
            if (sample.ts_end_ms < sample.ts_begin_ms || sample.ts_begin_ms < 0) {
                reasons.push_back("invalid_timestamp_order");
            } else {
                sample.sample_span_ms = static_cast<double>(sample.ts_end_ms - sample.ts_begin_ms);
                sample.ts_mid_ms = static_cast<double>(sample.ts_begin_ms) + 0.5 * sample.sample_span_ms;
            }
        }
        if (has_ac && has_usb && has_wireless && has_dock &&
            (sample.ac_powered || sample.usb_powered || sample.wireless_powered || sample.dock_powered)) {
            reasons.push_back("external_power_connected");
        }
        if (has_bat_status && sample.battery_status != cfg_.required_battery_status) {
            reasons.push_back("not_discharging");
        }
        if (has_bat_voltage &&
            !(sample.voltage_v >= cfg_.min_voltage_v && sample.voltage_v <= cfg_.max_voltage_v)) {
            reasons.push_back("voltage_out_of_range");
        }
        if (ibat.present) {
            sample.ibat_status = ibat.status;
            sample.current_a = (cfg_.ibat_unit == CurrentUnit::MilliAmps) ? (ibat.value / 1000.0) : ibat.value;
            if (sample.ibat_status != cfg_.required_ibat_status) reasons.push_back("ibat_sensor_fault");
            if (!(std::abs(sample.current_a) <= cfg_.max_abs_current_a)) reasons.push_back("current_out_of_range");
        }
        if ((has_pid && sample.pid <= 0) || (has_rss && sample.rss_kb < 0) || (has_vmhwm && sample.vmhwm_kb < 0)) {
            reasons.push_back("invalid_process_memory");
        }
        if (!reasons.empty()) {
            record_rejection(block_idx, start_line, reasons);
            return;
        }

        if (vbat.present) {
            sample.vbat_v = vbat.value / 1000.0;
            sample.has_vbat = true;
        }
        if (skin.present) {
            sample.skin_temp_c = skin.value;
            sample.has_skin_temp = true;
        }
        sample.power_w = sample.voltage_v * std::abs(sample.current_a);
        result->valid_samples.push_back(sample);
        ++result->valid_count;
    }
};

}  // namespace telemetry

#endif  // FRAMED_TELEMETRY_HPP
