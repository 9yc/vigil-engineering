#include "hdr_histogram.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cmath>
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

struct RequestRecord {
    double arrival_s = 0.0;
    double duration_ms = 0.0;
    std::string group = "default";
    std::string status = "ok";
};

struct CliOptions {
    std::string input_path, output_path, format = "auto", group_by = "group";
    double warmup_s = 0.0, window_s = 1.0, slo_ms = 100.0, max_trackable_ms = 3600000.0;
    int sigfigs = 3;
    std::vector<double> percentiles = {50.0, 90.0, 95.0, 99.0, 99.9};
};

std::string trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

double parse_strict_double(const std::string& raw, const std::string& name) {
    std::string s = trim(raw);
    if (s.empty()) throw std::runtime_error("Empty numeric token for " + name);
    for (unsigned char c : s) {
        char lo = static_cast<char>(std::tolower(c));
        if (lo == 'n' || lo == 'i') throw std::runtime_error("Non-finite token in " + name);
    }
    char* end = nullptr;
    errno = 0;
    double v = std::strtod(s.c_str(), &end);
    if (errno == ERANGE || end != s.c_str() + s.size() || !std::isfinite(v)) {
        throw std::runtime_error("Invalid float for " + name + ": " + s);
    }
    return v;
}

int parse_strict_int(const std::string& raw, const std::string& name) {
    std::string s = trim(raw);
    char* end = nullptr;
    errno = 0;
    long v = std::strtol(s.c_str(), &end, 10);
    if (s.empty() || errno == ERANGE || end != s.c_str() + s.size() || v < -100000L || v > 100000L) {
        throw std::runtime_error("Invalid integer for " + name);
    }
    return static_cast<int>(v);
}

std::vector<std::string> parse_csv_row(const std::string& line) {
    std::vector<std::string> cols;
    std::string cur;
    bool quoted = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (quoted) {
            if (c == '"' && i + 1 < line.size() && line[i + 1] == '"') { cur.push_back('"'); ++i; }
            else if (c == '"') quoted = false;
            else cur.push_back(c);
        } else if (c == '"') {
            if (!cur.empty()) throw std::runtime_error("Malformed CSV quote");
            quoted = true;
        } else if (c == ',') { cols.push_back(cur); cur.clear(); }
        else if (c != '\r') cur.push_back(c);
    }
    if (quoted) throw std::runtime_error("Unterminated CSV quote");
    cols.push_back(cur);
    return cols;
}

std::vector<RequestRecord> read_csv_records(std::istream& in) {
    std::string hdr_line;
    while (std::getline(in, hdr_line) && trim(hdr_line).empty()) {}
    if (trim(hdr_line).empty()) throw std::runtime_error("Empty CSV file");
    auto hdrs = parse_csv_row(hdr_line);
    int i_arr = -1, i_dur = -1, i_grp = -1, i_st = -1;
    for (size_t i = 0; i < hdrs.size(); ++i) {
        std::string h = trim(hdrs[i]);
        if (h == "arrival_s" || h == "start_s") i_arr = static_cast<int>(i);
        else if (h == "duration_ms" || h == "latency_ms") i_dur = static_cast<int>(i);
        else if (h == "group") i_grp = static_cast<int>(i);
        else if (h == "status") i_st = static_cast<int>(i);
    }
    if (i_arr < 0 || i_dur < 0) throw std::runtime_error("Missing required CSV latency/arrival columns");

    std::vector<RequestRecord> out;
    for (std::string line; std::getline(in, line);) {
        if (trim(line).empty()) continue;
        auto cols = parse_csv_row(line);
        if (cols.size() != hdrs.size()) throw std::runtime_error("CSV column count mismatch");
        RequestRecord r;
        r.arrival_s = parse_strict_double(cols[static_cast<size_t>(i_arr)], "arrival_s");
        r.duration_ms = parse_strict_double(cols[static_cast<size_t>(i_dur)], "duration_ms");
        if (r.arrival_s < 0.0 || r.duration_ms < 0.0) throw std::runtime_error("Negative arrival/latency");
        if (i_grp >= 0 && !trim(cols[static_cast<size_t>(i_grp)]).empty()) r.group = trim(cols[static_cast<size_t>(i_grp)]);
        if (i_st >= 0 && !trim(cols[static_cast<size_t>(i_st)]).empty()) r.status = trim(cols[static_cast<size_t>(i_st)]);
        out.push_back(std::move(r));
    }
    return out;
}

std::map<std::string, std::pair<bool, std::string>> parse_json_object(const std::string& s) {
    size_t pos = 0;
    auto skip = [&]() { while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) ++pos; };
    auto expect = [&](char c) {
        if (pos >= s.size() || s[pos] != c) throw std::runtime_error("Malformed JSON syntax");
        ++pos;
    };
    auto read_str = [&]() -> std::string {
        expect('"');
        std::string out;
        while (pos < s.size()) {
            char c = s[pos++];
            if (c == '"') return out;
            if (c == '\\') {
                if (pos >= s.size()) throw std::runtime_error("Bad JSON escape");
                char e = s[pos++];
                if (e == '"' || e == '\\' || e == '/') out.push_back(e);
                else if (e == 'n') out.push_back('\n');
                else if (e == 'r') out.push_back('\r');
                else if (e == 't') out.push_back('\t');
                else throw std::runtime_error("Unsupported escape");
            } else if (static_cast<unsigned char>(c) < 0x20) throw std::runtime_error("Control char in JSON");
            else out.push_back(c);
        }
        throw std::runtime_error("Unterminated JSON string");
    };

    skip(); expect('{'); skip();
    std::map<std::string, std::pair<bool, std::string>> kv;
    if (pos < s.size() && s[pos] == '}') {
        ++pos; skip();
        if (pos != s.size()) throw std::runtime_error("Trailing JSON tokens");
        return kv;
    }
    while (true) {
        skip(); std::string k = read_str(); skip(); expect(':'); skip();
        if (pos >= s.size()) throw std::runtime_error("Unexpected JSON EOF");
        if (s[pos] == '"') kv[k] = {true, read_str()};
        else {
            size_t b = pos;
            while (pos < s.size() && s[pos] != ',' && s[pos] != '}' &&
                   !std::isspace(static_cast<unsigned char>(s[pos]))) ++pos;
            std::string tok = s.substr(b, pos - b);
            if (tok.empty() || tok == "{" || tok == "[") throw std::runtime_error("Invalid JSON scalar");
            kv[k] = {false, tok};
        }
        skip();
        if (pos < s.size() && s[pos] == ',') { ++pos; continue; }
        expect('}');
        break;
    }
    skip();
    if (pos != s.size()) throw std::runtime_error("Trailing data after JSON object");
    return kv;
}

std::vector<RequestRecord> read_jsonl_records(std::istream& in) {
    std::vector<RequestRecord> out;
    for (std::string line; std::getline(in, line);) {
        std::string t = trim(line);
        if (t.empty()) continue;
        auto kv = parse_json_object(t);
        auto num = [&](const char* a, const char* b) {
            auto it = kv.find(a);
            if (it == kv.end()) it = kv.find(b);
            if (it == kv.end() || it->second.first) throw std::runtime_error(std::string("Missing ") + a);
            return parse_strict_double(it->second.second, a);
        };
        RequestRecord r;
        r.arrival_s = num("arrival_s", "start_s");
        r.duration_ms = num("duration_ms", "latency_ms");
        if (r.arrival_s < 0.0 || r.duration_ms < 0.0) throw std::runtime_error("Negative arrival/latency");
        if (auto it = kv.find("group"); it != kv.end() && it->second.second != "null") r.group = it->second.second;
        if (auto it = kv.find("status"); it != kv.end() && it->second.second != "null") r.status = it->second.second;
        out.push_back(std::move(r));
    }
    return out;
}

std::string json_escape(const std::string& s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\t') out += "\\t";
        else out.push_back(static_cast<char>(c));
    }
    return out + "\"";
}

bool is_ok_status(const std::string& st) {
    std::string lo;
    for (unsigned char c : st) lo.push_back(static_cast<char>(std::tolower(c)));
    return lo == "ok" || lo == "success" || lo == "200" || lo == "0";
}

void write_exclusive_file(const std::string& path, const std::string& body) {
    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (fd < 0) throw std::runtime_error("Refusing to clobber existing output: " + path);
    for (size_t off = 0; off < body.size();) {
        ssize_t n = ::write(fd, body.data() + off, body.size() - off);
        if (n < 0) { ::close(fd); throw std::runtime_error("Write failed: " + path); }
        off += static_cast<size_t>(n);
    }
    ::close(fd);
}

std::string render_stats_json(const std::vector<RequestRecord>& recs,
                              const CliOptions& opts,
                              bool include_cdf,
                              size_t indent) {
    hdr::DoubleHdrHistogram hist(opts.max_trackable_ms, opts.sigfigs, 1000.0, true);
    std::vector<hdr::RequestSpan> spans;
    spans.reserve(recs.size());
    size_t ok_cnt = 0, slo_cnt = 0;
    for (const auto& r : recs) {
        hist.record(r.duration_ms);
        spans.push_back({r.arrival_s, r.duration_ms});
        if (is_ok_status(r.status)) {
            ++ok_cnt;
            if (r.duration_ms <= opts.slo_ms) ++slo_cnt;
        }
    }
    auto win = hdr::analyze_sliding_window_and_littles_law(spans, opts.window_s);
    double denom = std::max(1e-9, hist.percentile(50.0));
    std::string p0(indent, ' '), p2(indent + 2, ' '), p4(indent + 4, ' ');
    std::ostringstream os;
    os << std::fixed << std::setprecision(6) << "{\n"
       << p2 << "\"count\": " << hist.total_count() << ",\n"
       << p2 << "\"error_count\": " << (recs.size() - ok_cnt) << ",\n"
       << p2 << "\"overflow_count\": " << hist.overflow_count() << ",\n"
       << p2 << "\"min_ms\": " << hist.min() << ",\n"
       << p2 << "\"max_ms\": " << hist.max() << ",\n"
       << p2 << "\"mean_ms\": " << hist.mean() << ",\n"
       << p2 << "\"stddev_ms\": " << hist.stddev() << ",\n"
       << p2 << "\"percentiles_ms\": {\n";
    for (size_t i = 0; i < opts.percentiles.size(); ++i) {
        std::ostringstream pk;
        pk << "P" << std::noshowpoint << std::setprecision(6) << opts.percentiles[i];
        os << p4 << json_escape(pk.str()) << ": " << hist.percentile(opts.percentiles[i])
           << (i + 1 < opts.percentiles.size() ? "," : "") << "\n";
    }
    os << p2 << "},\n"
       << p2 << "\"tail_to_median\": {\"p95_to_p50\": " << (hist.percentile(95.0) / denom)
       << ", \"p99_to_p50\": " << (hist.percentile(99.0) / denom)
       << ", \"p999_to_p50\": " << (hist.percentile(99.9) / denom) << "},\n"
       << p2 << "\"throughput\": {\"span_s\": " << win.span_s
       << ", \"mean_rps\": " << win.mean_arrival_rps
       << ", \"window_s\": " << win.window_s
       << ", \"peak_arrival_rps\": " << win.peak_arrival_window_rps
       << ", \"peak_completion_rps\": " << win.peak_completion_window_rps << "},\n"
       << p2 << "\"concurrency\": {\"littles_law_mean\": " << win.littles_law_concurrency
       << ", \"time_weighted_mean\": " << win.time_weighted_concurrency
       << ", \"peak_active\": " << win.peak_concurrency << "},\n"
       << p2 << "\"slo\": {\"threshold_ms\": " << opts.slo_ms
       << ", \"attained_count\": " << slo_cnt
       << ", \"attainment_ratio\": " << (static_cast<double>(slo_cnt) / static_cast<double>(recs.size())) << "}";
    if (include_cdf) {
        auto cdf = hist.export_cdf(8);
        os << ",\n" << p2 << "\"cdf\": [";
        for (size_t i = 0; i < cdf.size(); ++i) {
            os << (i ? ", " : "") << "{\"low_ms\": " << cdf[i].value_low << ", \"high_ms\": " << cdf[i].value_high
               << ", \"count\": " << cdf[i].count << ", \"cdf\": " << cdf[i].cdf << "}";
        }
        os << "]";
    }
    os << "\n" << p0 << "}";
    return os.str();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        CliOptions opts;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::runtime_error("Missing value after " + a);
                return argv[++i];
            };
            if (a == "--input" || a == "-i") opts.input_path = next();
            else if (a == "--output" || a == "-o") opts.output_path = next();
            else if (a == "--format") opts.format = next();
            else if (a == "--warmup") opts.warmup_s = parse_strict_double(next(), "--warmup");
            else if (a == "--window-s") opts.window_s = parse_strict_double(next(), "--window-s");
            else if (a == "--slo-ms") opts.slo_ms = parse_strict_double(next(), "--slo-ms");
            else if (a == "--max-trackable-ms") opts.max_trackable_ms = parse_strict_double(next(), "--max-trackable-ms");
            else if (a == "--sigfigs") opts.sigfigs = parse_strict_int(next(), "--sigfigs");
            else if (a == "--group-by") opts.group_by = next();
            else if (a == "--percentiles") {
                opts.percentiles.clear();
                std::stringstream ss(next());
                for (std::string tok; std::getline(ss, tok, ',');) opts.percentiles.push_back(parse_strict_double(tok, "pct"));
            } else if (!a.empty() && a[0] != '-' && opts.input_path.empty()) opts.input_path = a;
            else throw std::runtime_error("Unknown option: " + a);
        }
        if (opts.input_path.empty() || opts.warmup_s < 0.0 || opts.window_s <= 0.0 || opts.slo_ms <= 0.0 ||
            opts.percentiles.empty() ||
            (opts.group_by != "group" && opts.group_by != "status" && opts.group_by != "none")) {
            throw std::runtime_error("Invalid CLI arguments");
        }

        std::ifstream fin(opts.input_path);
        if (!fin) throw std::runtime_error("Cannot open input: " + opts.input_path);
        std::string fmt = opts.format == "auto"
                              ? ((opts.input_path.size() >= 6 && opts.input_path.substr(opts.input_path.size() - 6) == ".jsonl")
                                     ? "jsonl" : "csv")
                              : opts.format;
        auto raw = (fmt == "jsonl") ? read_jsonl_records(fin)
                                    : (fmt == "csv" ? read_csv_records(fin) : throw std::runtime_error("Unsupported format"));
        if (raw.empty()) throw std::runtime_error("Zero request records found");

        double t0 = raw.front().arrival_s;
        for (const auto& r : raw) t0 = std::min(t0, r.arrival_s);
        std::vector<RequestRecord> active;
        for (const auto& r : raw) if (r.arrival_s + 1e-12 >= t0 + opts.warmup_s) active.push_back(r);
        if (active.empty()) throw std::runtime_error("All records removed by warmup");

        std::map<std::string, std::vector<RequestRecord>> groups;
        if (opts.group_by != "none") {
            for (const auto& r : active) groups[opts.group_by == "status" ? r.status : r.group].push_back(r);
        }

        std::ostringstream doc;
        doc << "{\n  \"warmup_skipped\": " << (raw.size() - active.size())
            << ",\n  \"overall\": " << render_stats_json(active, opts, true, 2) << ",\n  \"groups\": {";
        size_t idx = 0;
        for (const auto& [k, v] : groups) {
            doc << (idx++ ? ",\n" : "\n") << "    " << json_escape(k) << ": " << render_stats_json(v, opts, false, 4);
        }
        doc << (groups.empty() ? "}\n}\n" : "\n  }\n}\n");

        if (opts.output_path.empty()) std::cout << doc.str();
        else write_exclusive_file(opts.output_path, doc.str());
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "latency_aggregator error: " << ex.what() << "\n";
        return 1;
    }
}
