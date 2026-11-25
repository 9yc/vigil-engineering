#include "framed_log.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string trim(const std::string& s) {
    size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

uint64_t parse_u64(const std::string& raw, const char* name) {
    std::string s = trim(raw);
    if (s.empty() || s[0] == '-') {
        throw std::runtime_error(std::string("Invalid uint64 for ") + name);
    }
    char* end = nullptr;
    errno = 0;
    unsigned long long v = std::strtoull(s.c_str(), &end, 10);
    if (errno != 0 || end != s.c_str() + s.size()) {
        throw std::runtime_error(std::string("Malformed integer for ") + name + ": " + s);
    }
    return static_cast<uint64_t>(v);
}

uint16_t parse_u16(const std::string& raw, const char* name) {
    uint64_t v = parse_u64(raw, name);
    if (v > 65535ULL) {
        throw std::runtime_error(std::string("Value exceeds uint16 for ") + name);
    }
    return static_cast<uint16_t>(v);
}

std::string json_escape(const std::string& s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
                break;
        }
    }
    out.push_back('"');
    return out;
}

void write_exclusive_text(const std::string& path, const std::string& text) {
    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (fd < 0) {
        throw std::runtime_error("Refusing to clobber or failed creating output: " + path);
    }
    size_t off = 0;
    while (off < text.size()) {
        ssize_t n = ::write(fd, text.data() + off, text.size() - off);
        if (n < 0) {
            ::close(fd);
            throw std::runtime_error("Failed writing output: " + path);
        }
        off += static_cast<size_t>(n);
    }
    ::close(fd);
}

class JsonLineReader {
public:
    explicit JsonLineReader(const std::string& line) : s_(line) {}

    std::map<std::string, std::pair<bool, std::string>> parse() {
        skip_ws();
        expect('{');
        skip_ws();
        std::map<std::string, std::pair<bool, std::string>> res;
        if (peek() == '}') {
            ++pos_;
            skip_ws();
            if (pos_ != s_.size()) throw std::runtime_error("Trailing garbage after '}'");
            return res;
        }
        while (true) {
            skip_ws();
            std::string k = parse_str();
            skip_ws();
            expect(':');
            skip_ws();
            res[k] = parse_val();
            skip_ws();
            if (peek() == ',') {
                ++pos_;
                continue;
            }
            if (peek() == '}') {
                ++pos_;
                break;
            }
            throw std::runtime_error("Expected ',' or '}' in JSON record");
        }
        skip_ws();
        if (pos_ != s_.size()) throw std::runtime_error("Trailing tokens on JSONL line");
        return res;
    }

private:
    char peek() const {
        if (pos_ >= s_.size()) throw std::runtime_error("Unexpected EOF in JSONL line");
        return s_[pos_];
    }

    void expect(char c) {
        if (peek() != c) throw std::runtime_error(std::string("Expected '") + c + "'");
        ++pos_;
    }

    void skip_ws() {
        while (pos_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[pos_]))) ++pos_;
    }

    std::string parse_str() {
        expect('"');
        std::string out;
        while (pos_ < s_.size()) {
            char c = s_[pos_++];
            if (c == '"') return out;
            if (c == '\\') {
                if (pos_ >= s_.size()) throw std::runtime_error("Bad JSON escape");
                char e = s_[pos_++];
                switch (e) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    default: throw std::runtime_error("Unsupported JSON escape");
                }
            } else {
                out.push_back(c);
            }
        }
        throw std::runtime_error("Unterminated JSON string");
    }

    std::pair<bool, std::string> parse_val() {
        if (peek() == '"') {
            return {true, parse_str()};
        }
        size_t b = pos_;
        while (pos_ < s_.size() && s_[pos_] != ',' && s_[pos_] != '}' &&
               !std::isspace(static_cast<unsigned char>(s_[pos_]))) {
            ++pos_;
        }
        std::string tok = s_.substr(b, pos_ - b);
        if (tok.empty()) throw std::runtime_error("Empty JSON scalar");
        return {false, tok};
    }

    const std::string& s_;
    size_t pos_ = 0;
};

std::string format_records_jsonl(const std::vector<flog::LogRecord>& recs) {
    std::ostringstream os;
    for (const auto& r : recs) {
        os << "{\"seq\": " << r.seq
           << ", \"timestamp_ns\": " << r.timestamp_ns
           << ", \"type\": " << r.record_type
           << ", \"payload\": " << json_escape(r.payload) << "}\n";
    }
    return os.str();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            throw std::runtime_error("Usage: trace_spooler <ingest|verify|query|export> [options]");
        }
        std::string cmd = argv[1];
        std::string input_path, log_dir, output_path;
        size_t max_seg_bytes = 64 * 1024;
        size_t max_seg_records = 4096;
        uint16_t index_stride = 8;
        uint64_t initial_seq = 1;
        bool recover_tail = false;
        std::optional<uint64_t> seq_start, seq_end;
        std::optional<uint16_t> type_filter;

        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::runtime_error("Missing value for " + a);
                return argv[++i];
            };
            if (a == "--input" || a == "-i") input_path = next();
            else if (a == "--log-dir" || a == "-d") log_dir = next();
            else if (a == "--output" || a == "-o") output_path = next();
            else if (a == "--max-segment-bytes") max_seg_bytes = static_cast<size_t>(parse_u64(next(), "--max-segment-bytes"));
            else if (a == "--max-segment-records") max_seg_records = static_cast<size_t>(parse_u64(next(), "--max-segment-records"));
            else if (a == "--index-stride") index_stride = parse_u16(next(), "--index-stride");
            else if (a == "--initial-seq") initial_seq = parse_u64(next(), "--initial-seq");
            else if (a == "--seq-start") seq_start = parse_u64(next(), "--seq-start");
            else if (a == "--seq-end") seq_end = parse_u64(next(), "--seq-end");
            else if (a == "--type") type_filter = parse_u16(next(), "--type");
            else if (a == "--recover-tail") recover_tail = true;
            else throw std::runtime_error("Unknown flag: " + a);
        }

        if (log_dir.empty()) {
            throw std::runtime_error("Missing required --log-dir <dir>");
        }

        if (cmd == "ingest") {
            if (input_path.empty()) throw std::runtime_error("ingest requires --input <jsonl>");
            if (std::filesystem::exists(log_dir) && !std::filesystem::is_empty(log_dir)) {
                throw std::runtime_error("Refusing to ingest into non-empty log-dir: " + log_dir);
            }
            std::ifstream in(input_path);
            if (!in) throw std::runtime_error("Cannot open JSONL input: " + input_path);

            flog::WriterConfig cfg{max_seg_bytes, max_seg_records, index_stride, initial_seq};
            flog::FramedLogWriter writer(log_dir, cfg);

            std::string line;
            uint64_t count = 0;
            uint64_t first_seq = 0, last_seq = 0;
            while (std::getline(in, line)) {
                std::string t = trim(line);
                if (t.empty()) continue;
                auto kv = JsonLineReader(t).parse();
                auto ts_it = kv.find("timestamp_ns");
                auto ty_it = kv.find("type");
                auto py_it = kv.find("payload");
                if (ts_it == kv.end() || ts_it->second.first ||
                    ty_it == kv.end() || ty_it->second.first ||
                    py_it == kv.end() || !py_it->second.first) {
                    throw std::runtime_error("JSONL record requires numeric timestamp_ns, numeric type, string payload");
                }
                uint64_t ts = parse_u64(ts_it->second.second, "timestamp_ns");
                uint16_t rtype = parse_u16(ty_it->second.second, "type");
                if (auto sq_it = kv.find("seq"); sq_it != kv.end()) {
                    if (sq_it->second.first || parse_u64(sq_it->second.second, "seq") != writer.next_sequence()) {
                        throw std::runtime_error("Explicit seq violates monotonic sequence order");
                    }
                }
                uint64_t seq = writer.append(ts, rtype, py_it->second.second);
                if (count == 0) first_seq = seq;
                last_seq = seq;
                ++count;
            }
            if (count == 0) throw std::runtime_error("No records found in JSONL input");
            writer.flush();

            std::ostringstream js;
            js << "{\n  \"records_ingested\": " << count << ",\n"
               << "  \"segments_created\": " << writer.segments_created() << ",\n"
               << "  \"first_seq\": " << first_seq << ",\n"
               << "  \"last_seq\": " << last_seq << ",\n"
               << "  \"total_bytes\": " << writer.total_bytes() << "\n}\n";
            if (!output_path.empty()) write_exclusive_text(output_path, js.str());
            else std::cout << js.str();
        } else if (cmd == "verify") {
            auto rep = flog::verify_log_directory(log_dir, recover_tail);
            std::ostringstream js;
            js << "{\n  \"valid\": " << (rep.valid ? "true" : "false") << ",\n"
               << "  \"segments_checked\": " << rep.segments_checked << ",\n"
               << "  \"records_verified\": " << rep.records_verified << ",\n"
               << "  \"corrupt_frames\": " << rep.corrupt_frames << ",\n"
               << "  \"truncated_tail_detected\": " << (rep.truncated_tail_detected ? "true" : "false") << ",\n"
               << "  \"recovered_tail\": " << (rep.recovered_tail ? "true" : "false") << ",\n"
               << "  \"truncated_bytes_removed\": " << rep.truncated_bytes_removed << ",\n"
               << "  \"error_detail\": " << json_escape(rep.error_detail) << "\n}\n";
            if (!output_path.empty()) write_exclusive_text(output_path, js.str());
            else std::cout << js.str();
            return rep.valid ? 0 : 1;
        } else if (cmd == "query") {
            if (seq_start.has_value() && seq_end.has_value() && *seq_start > *seq_end) {
                throw std::runtime_error("--seq-start cannot exceed --seq-end");
            }
            auto recs = flog::query_records(log_dir, seq_start, seq_end, type_filter);
            std::string out = format_records_jsonl(recs);
            if (!output_path.empty()) write_exclusive_text(output_path, out);
            else std::cout << out;
        } else if (cmd == "export") {
            if (output_path.empty()) throw std::runtime_error("export requires --output <jsonl>");
            auto rep = flog::verify_log_directory(log_dir, false);
            if (!rep.valid) {
                throw std::runtime_error("Refusing to export unverified/corrupt log: " + rep.error_detail);
            }
            auto recs = flog::query_records(log_dir, std::nullopt, std::nullopt, std::nullopt);
            write_exclusive_text(output_path, format_records_jsonl(recs));
        } else {
            throw std::runtime_error("Unknown subcommand: " + cmd);
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "trace_spooler error: " << ex.what() << "\n";
        return 1;
    }
}
