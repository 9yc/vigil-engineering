#include "framed_telemetry.hpp"

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

enum class OutputFormat { Auto, Jsonl, Csv };

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

void write_exclusive_file(const std::string& path, const std::string& content) {
    if (path.empty()) throw std::invalid_argument("output path must not be empty");
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (fd < 0) {
        throw std::runtime_error("exclusive open failed for '" + path + "': " + std::strerror(errno));
    }
    std::size_t written = 0;
    while (written < content.size()) {
        const ssize_t rc = ::write(fd, content.data() + written, content.size() - written);
        if (rc < 0) {
            if (errno == EINTR) continue;
            const int err = errno;
            ::close(fd);
            throw std::runtime_error("write failed for '" + path + "': " + std::strerror(err));
        }
        written += static_cast<std::size_t>(rc);
    }
    if (::close(fd) != 0) {
        throw std::runtime_error("close failed for '" + path + "': " + std::strerror(errno));
    }
}

std::string read_entire_file(const std::string& path) {
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs.is_open()) throw std::runtime_error("unable to open input stream: " + path);
    std::ostringstream buffer;
    buffer << ifs.rdbuf();
    return buffer.str();
}

double parse_double_flag(const std::string& flag, const std::string& raw) {
    std::size_t idx = 0;
    double val = 0.0;
    try {
        val = std::stod(raw, &idx);
    } catch (...) {
        throw std::invalid_argument("invalid float for " + flag + ": " + raw);
    }
    if (idx != raw.size() || !std::isfinite(val)) {
        throw std::invalid_argument("invalid float for " + flag + ": " + raw);
    }
    return val;
}

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string format_samples_jsonl(const std::vector<telemetry::TelemetrySample>& samples) {
    std::ostringstream oss;
    oss << std::setprecision(10);
    for (const auto& s : samples) {
        oss << "{\"block_index\": " << s.block_index
            << ", \"ts_begin_ms\": " << s.ts_begin_ms
            << ", \"ts_end_ms\": " << s.ts_end_ms
            << ", \"ts_mid_ms\": " << std::fixed << std::setprecision(1) << s.ts_mid_ms
            << std::defaultfloat << std::setprecision(10)
            << ", \"sample_span_ms\": " << s.sample_span_ms
            << ", \"voltage_v\": " << s.voltage_v
            << ", \"current_a\": " << s.current_a
            << ", \"power_w\": " << s.power_w
            << ", \"battery_status\": " << s.battery_status
            << ", \"battery_level\": " << s.battery_level
            << ", \"battery_temp_c\": " << s.battery_temp_c
            << ", \"ibat_status\": " << s.ibat_status
            << ", \"vbat_v\": " << (s.has_vbat ? std::to_string(s.vbat_v) : "null")
            << ", \"skin_temp_c\": " << (s.has_skin_temp ? std::to_string(s.skin_temp_c) : "null")
            << ", \"pid\": " << s.pid
            << ", \"rss_kb\": " << s.rss_kb
            << ", \"vmhwm_kb\": " << s.vmhwm_kb << "}\n";
    }
    return oss.str();
}

std::string format_samples_csv(const std::vector<telemetry::TelemetrySample>& samples) {
    std::ostringstream oss;
    oss << "block_index,ts_begin_ms,ts_end_ms,ts_mid_ms,sample_span_ms,"
           "voltage_v,current_a,power_w,battery_status,battery_level,battery_temp_c,"
           "ibat_status,vbat_v,skin_temp_c,pid,rss_kb,vmhwm_kb\n";
    for (const auto& s : samples) {
        oss << s.block_index << "," << s.ts_begin_ms << "," << s.ts_end_ms << ","
            << std::fixed << std::setprecision(1) << s.ts_mid_ms << ","
            << std::defaultfloat << std::setprecision(10)
            << s.sample_span_ms << "," << s.voltage_v << "," << s.current_a << ","
            << s.power_w << "," << s.battery_status << "," << s.battery_level << ","
            << s.battery_temp_c << "," << s.ibat_status << ",";
        if (s.has_vbat) oss << s.vbat_v;
        oss << ",";
        if (s.has_skin_temp) oss << s.skin_temp_c;
        oss << "," << s.pid << "," << s.rss_kb << "," << s.vmhwm_kb << "\n";
    }
    return oss.str();
}

std::string format_summary_json(
    const telemetry::StreamDecodeResult& res, const telemetry::DecoderConfig& cfg) {
    std::ostringstream oss;
    oss << std::setprecision(10)
        << "{\n"
        << "  \"allow_partial_tail\": " << (cfg.allow_partial_tail ? "true" : "false") << ",\n"
        << "  \"total_blocks_seen\": " << res.total_blocks_seen << ",\n"
        << "  \"complete_blocks\": " << res.complete_blocks << ",\n"
        << "  \"partial_tail_blocks\": " << res.partial_tail_blocks << ",\n"
        << "  \"valid_samples\": " << res.valid_count << ",\n"
        << "  \"rejected_samples\": " << res.rejected_count << ",\n"
        << "  \"mean_voltage_v\": " << res.mean_voltage_v << ",\n"
        << "  \"mean_abs_current_a\": " << res.mean_abs_current_a << ",\n"
        << "  \"mean_power_w\": " << res.mean_power_w << ",\n"
        << "  \"max_skin_temp_c\": " << res.max_skin_temp_c << ",\n"
        << "  \"max_rss_kb\": " << res.max_rss_kb << ",\n"
        << "  \"max_vmhwm_kb\": " << res.max_vmhwm_kb << ",\n"
        << "  \"rejected_by_reason\": {\n";
    std::size_t idx = 0;
    for (const auto& kv : res.rejected_by_reason) {
        oss << "    \"" << json_escape(kv.first) << "\": " << kv.second
            << (++idx < res.rejected_by_reason.size() ? "," : "") << "\n";
    }
    oss << "  },\n  \"rejected_blocks\": [\n";
    for (std::size_t i = 0; i < res.rejected_blocks.size(); ++i) {
        const auto& rb = res.rejected_blocks[i];
        oss << "    {\"block_index\": " << rb.block_index
            << ", \"start_line\": " << rb.start_line
            << ", \"primary_reason\": \"" << json_escape(rb.primary_reason)
            << "\", \"reasons\": [";
        for (std::size_t j = 0; j < rb.all_reasons.size(); ++j) {
            oss << "\"" << json_escape(rb.all_reasons[j]) << "\""
                << (j + 1 < rb.all_reasons.size() ? ", " : "");
        }
        oss << "]}" << (i + 1 < res.rejected_blocks.size() ? "," : "") << "\n";
    }
    oss << "  ]\n}\n";
    return oss.str();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string input_path;
        std::string output_path;
        std::string summary_path;
        OutputFormat fmt = OutputFormat::Auto;
        telemetry::DecoderConfig cfg;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto next_val = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument("missing value after flag " + arg);
                return argv[++i];
            };
            if (arg == "--input") input_path = next_val();
            else if (arg == "--output") output_path = next_val();
            else if (arg == "--summary-output") summary_path = next_val();
            else if (arg == "--allow-partial-tail") cfg.allow_partial_tail = true;
            else if (arg == "--format") {
                const std::string f = next_val();
                if (f == "jsonl") fmt = OutputFormat::Jsonl;
                else if (f == "csv") fmt = OutputFormat::Csv;
                else throw std::invalid_argument("unsupported --format: " + f);
            } else if (arg == "--ibat-unit") {
                const std::string u = next_val();
                if (u == "ma" || u == "mA") cfg.ibat_unit = telemetry::CurrentUnit::MilliAmps;
                else if (u == "a" || u == "A") cfg.ibat_unit = telemetry::CurrentUnit::Amps;
                else throw std::invalid_argument("unsupported --ibat-unit: " + u);
            } else if (arg == "--min-voltage-v") cfg.min_voltage_v = parse_double_flag(arg, next_val());
            else if (arg == "--max-voltage-v") cfg.max_voltage_v = parse_double_flag(arg, next_val());
            else if (arg == "--max-abs-current-a") cfg.max_abs_current_a = parse_double_flag(arg, next_val());
            else throw std::invalid_argument("unknown CLI option: " + arg);
        }

        if (input_path.empty() || output_path.empty()) {
            throw std::invalid_argument("both --input and --output are required");
        }
        if (!summary_path.empty() && summary_path == output_path) {
            throw std::invalid_argument("--output and --summary-output must be distinct paths");
        }

        cfg.validate();
        const std::string raw_stream = read_entire_file(input_path);
        telemetry::TelemetryBlockDecoder decoder(cfg);
        const telemetry::StreamDecodeResult result = decoder.decode_stream(raw_stream);

        if (fmt == OutputFormat::Auto) {
            fmt = ends_with(output_path, ".csv") ? OutputFormat::Csv : OutputFormat::Jsonl;
        }
        const std::string payload = (fmt == OutputFormat::Csv)
                                        ? format_samples_csv(result.valid_samples)
                                        : format_samples_jsonl(result.valid_samples);
        write_exclusive_file(output_path, payload);
        if (!summary_path.empty()) {
            write_exclusive_file(summary_path, format_summary_json(result, cfg));
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "telemetry_block_decoder error: " << ex.what() << "\n";
        return 1;
    }
}
