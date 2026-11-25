#include "quant_pack.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using qpack::ScaleDtype;
using qpack::TensorDtype;
using qpack::TensorHeader;

struct DecodedTensor {
    TensorHeader header;
    size_t total_file_bytes = 0;
    std::vector<float> values;
};

std::string dtype_to_name(TensorDtype dt) {
    switch (dt) {
        case TensorDtype::F32: return "f32";
        case TensorDtype::F16: return "f16";
        case TensorDtype::BF16: return "bf16";
        case TensorDtype::Q8_SYM: return "q8_sym";
        case TensorDtype::Q8_ASYM: return "q8_asym";
        case TensorDtype::Q4_SYM: return "q4_sym";
        case TensorDtype::Q4_ASYM: return "q4_asym";
    }
    throw std::runtime_error("Invalid TensorDtype");
}

TensorDtype parse_dtype(const std::string& s) {
    if (s == "f32") return TensorDtype::F32;
    if (s == "f16") return TensorDtype::F16;
    if (s == "bf16") return TensorDtype::BF16;
    if (s == "q8_sym") return TensorDtype::Q8_SYM;
    if (s == "q8_asym") return TensorDtype::Q8_ASYM;
    if (s == "q4_sym") return TensorDtype::Q4_SYM;
    if (s == "q4_asym") return TensorDtype::Q4_ASYM;
    throw std::runtime_error("Unsupported dtype: " + s);
}

bool is_quantized(TensorDtype dt) { return static_cast<uint8_t>(dt) >= 3u; }
bool is_asymmetric(TensorDtype dt) { return dt == TensorDtype::Q8_ASYM || dt == TensorDtype::Q4_ASYM; }

uint64_t expected_payload_bytes(TensorDtype dt, uint64_t n) {
    if (dt == TensorDtype::F32) return n * 4ULL;
    if (dt == TensorDtype::F16 || dt == TensorDtype::BF16) return n * 2ULL;
    if (dt == TensorDtype::Q8_SYM || dt == TensorDtype::Q8_ASYM) return n;
    return (n + 1ULL) / 2ULL;
}

std::vector<uint8_t> read_all_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("Failed to open: " + path);
    auto sz = in.tellg();
    if (sz < 0) throw std::runtime_error("Cannot stat: " + path);
    std::vector<uint8_t> buf(static_cast<size_t>(sz));
    in.seekg(0);
    if (!buf.empty() && !in.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()))) {
        throw std::runtime_error("Short read: " + path);
    }
    return buf;
}

void write_exclusive_bytes(const std::string& path, const void* data, size_t len) {
    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (fd < 0) throw std::runtime_error("Output exists or cannot create: " + path);
    const auto* ptr = static_cast<const uint8_t*>(data);
    for (size_t off = 0; off < len;) {
        ssize_t n = ::write(fd, ptr + off, len - off);
        if (n < 0) { ::close(fd); throw std::runtime_error("Write failure: " + path); }
        off += static_cast<size_t>(n);
    }
    ::close(fd);
}

DecodedTensor decode_qtns_buffer(const std::vector<uint8_t>& buf) {
    if (buf.size() < sizeof(TensorHeader)) throw std::runtime_error("Truncated QTNS header");
    TensorHeader hdr;
    std::memcpy(&hdr, buf.data(), sizeof(TensorHeader));
    if (std::memcmp(hdr.magic, "QTNS", 4) != 0 || hdr.version != 1 ||
        hdr.dtype > 6u || hdr.scale_dtype > 2u || hdr.rows == 0 || hdr.cols == 0) {
        throw std::runtime_error("Invalid QTNS header metadata");
    }
    uint64_t total_elems = static_cast<uint64_t>(hdr.rows) * static_cast<uint64_t>(hdr.cols);
    if (total_elems > 200000000ULL) throw std::runtime_error("Element count exceeds limit");

    TensorDtype dt = static_cast<TensorDtype>(hdr.dtype);
    ScaleDtype sdt = static_cast<ScaleDtype>(hdr.scale_dtype);
    uint64_t exp_payload = expected_payload_bytes(dt, total_elems);
    if (hdr.payload_bytes != exp_payload) throw std::runtime_error("Header payload_bytes mismatch");

    uint64_t scale_bytes = 0, zp_bytes = 0;
    if (is_quantized(dt)) {
        if (hdr.group_size == 0 || sdt == ScaleDtype::NONE) throw std::runtime_error("Invalid quant params");
        uint64_t exp_groups = (total_elems + hdr.group_size - 1ULL) / hdr.group_size;
        if (hdr.num_groups != exp_groups) throw std::runtime_error("Header num_groups mismatch");
        scale_bytes = exp_groups * (sdt == ScaleDtype::F32 ? 4ULL : 2ULL);
        zp_bytes = is_asymmetric(dt) ? exp_groups : 0ULL;
    } else if (hdr.num_groups != 0 || sdt != ScaleDtype::NONE) {
        throw std::runtime_error("Unquantized tensor has unexpected scale metadata");
    }

    if (buf.size() != sizeof(TensorHeader) + scale_bytes + zp_bytes + exp_payload) {
        throw std::runtime_error("Corrupted or truncated QTNS container size");
    }

    size_t cur = sizeof(TensorHeader);
    std::vector<float> scales(hdr.num_groups, 0.0f);
    if (hdr.num_groups > 0) {
        if (sdt == ScaleDtype::F32) {
            std::memcpy(scales.data(), buf.data() + cur, static_cast<size_t>(scale_bytes));
        } else {
            std::vector<uint16_t> hs(hdr.num_groups);
            std::memcpy(hs.data(), buf.data() + cur, static_cast<size_t>(scale_bytes));
            scales = qpack::fp16_span_to_fp32(hs.data(), hs.size());
        }
        for (float s : scales) {
            if (!std::isfinite(s) || s < 0.0f) throw std::runtime_error("Corrupt scale value in QTNS");
        }
        cur += static_cast<size_t>(scale_bytes);
    }

    std::vector<uint8_t> zps(hdr.num_groups, 0u);
    if (zp_bytes > 0) {
        std::memcpy(zps.data(), buf.data() + cur, static_cast<size_t>(zp_bytes));
        cur += static_cast<size_t>(zp_bytes);
    }

    const uint8_t* pay = buf.data() + cur;
    size_t n = static_cast<size_t>(total_elems);
    DecodedTensor out{hdr, buf.size(), std::vector<float>(n, 0.0f)};
    if (dt == TensorDtype::F32) {
        std::memcpy(out.values.data(), pay, n * 4);
    } else if (dt == TensorDtype::F16 || dt == TensorDtype::BF16) {
        std::vector<uint16_t> words(n);
        std::memcpy(words.data(), pay, n * 2);
        out.values = (dt == TensorDtype::F16) ? qpack::fp16_span_to_fp32(words.data(), n)
                                              : qpack::bf16_span_to_fp32(words.data(), n);
    } else if (dt == TensorDtype::Q8_SYM) {
        qpack::SymQuant8Block b{hdr.group_size, n, std::move(scales), std::vector<int8_t>(n, 0)};
        std::memcpy(b.data.data(), pay, n);
        out.values = qpack::dequantize_q8_symmetric(b);
    } else if (dt == TensorDtype::Q8_ASYM) {
        out.values = qpack::dequantize_q8_asymmetric(
            {hdr.group_size, n, std::move(scales), std::move(zps), std::vector<uint8_t>(pay, pay + n)});
    } else if (dt == TensorDtype::Q4_SYM) {
        out.values = qpack::dequantize_q4_symmetric(
            {hdr.group_size, n, std::move(scales), std::vector<uint8_t>(pay, pay + exp_payload)});
    } else {
        out.values = qpack::dequantize_q4_asymmetric(
            {hdr.group_size, n, std::move(scales), std::move(zps), std::vector<uint8_t>(pay, pay + exp_payload)});
    }
    return out;
}

DecodedTensor load_source(const std::string& path, uint32_t rows, uint32_t cols) {
    auto buf = read_all_bytes(path);
    if (buf.size() >= 4 && std::memcmp(buf.data(), "QTNS", 4) == 0) {
        auto dec = decode_qtns_buffer(buf);
        if (rows != 0 && (dec.header.rows != rows || dec.header.cols != cols)) {
            throw std::runtime_error("CLI shape mismatches QTNS header");
        }
        return dec;
    }
    if (rows == 0 || cols == 0) throw std::runtime_error("Raw f32 requires --rows and --cols");
    uint64_t elems = static_cast<uint64_t>(rows) * static_cast<uint64_t>(cols);
    if (buf.size() != elems * 4ULL) throw std::runtime_error("Raw f32 byte size mismatch");
    DecodedTensor out;
    out.header.rows = rows;
    out.header.cols = cols;
    out.header.payload_bytes = elems * 4ULL;
    out.total_file_bytes = buf.size();
    out.values.resize(static_cast<size_t>(elems));
    std::memcpy(out.values.data(), buf.data(), buf.size());
    return out;
}

std::vector<uint8_t> encode_qtns(const std::vector<float>& src,
                                 uint32_t rows,
                                 uint32_t cols,
                                 TensorDtype dt,
                                 uint32_t group_size,
                                 ScaleDtype sdt) {
    size_t n = src.size();
    TensorHeader hdr;
    hdr.dtype = static_cast<uint8_t>(dt);
    hdr.rows = rows;
    hdr.cols = cols;
    hdr.payload_bytes = expected_payload_bytes(dt, n);

    std::vector<float> scales;
    std::vector<uint8_t> zps, payload(static_cast<size_t>(hdr.payload_bytes), 0u);
    bool fp16_s = (sdt == ScaleDtype::F16);

    if (dt == TensorDtype::F32) {
        for (float v : src) if (!std::isfinite(v)) throw std::runtime_error("Non-finite f32 rejected");
        std::memcpy(payload.data(), src.data(), payload.size());
    } else if (dt == TensorDtype::F16 || dt == TensorDtype::BF16) {
        auto words = (dt == TensorDtype::F16) ? qpack::fp32_span_to_fp16(src.data(), n)
                                              : qpack::fp32_span_to_bf16(src.data(), n);
        std::memcpy(payload.data(), words.data(), payload.size());
    } else if (dt == TensorDtype::Q8_SYM) {
        auto b = qpack::quantize_q8_symmetric(src.data(), n, group_size, fp16_s);
        scales = std::move(b.scales);
        std::memcpy(payload.data(), b.data.data(), n);
    } else if (dt == TensorDtype::Q8_ASYM) {
        auto b = qpack::quantize_q8_asymmetric(src.data(), n, group_size, fp16_s);
        scales = std::move(b.scales); zps = std::move(b.zero_points); payload = std::move(b.data);
    } else if (dt == TensorDtype::Q4_SYM) {
        auto b = qpack::quantize_q4_symmetric(src.data(), n, group_size, fp16_s);
        scales = std::move(b.scales); payload = std::move(b.packed_data);
    } else {
        auto b = qpack::quantize_q4_asymmetric(src.data(), n, group_size, fp16_s);
        scales = std::move(b.scales); zps = std::move(b.zero_points); payload = std::move(b.packed_data);
    }

    if (is_quantized(dt)) {
        hdr.scale_dtype = static_cast<uint8_t>(sdt);
        hdr.group_size = group_size;
        hdr.num_groups = static_cast<uint32_t>(scales.size());
    }

    std::vector<uint8_t> out(sizeof(TensorHeader));
    std::memcpy(out.data(), &hdr, sizeof(TensorHeader));
    if (sdt == ScaleDtype::F32 && !scales.empty()) {
        size_t p = out.size();
        out.resize(p + scales.size() * 4);
        std::memcpy(out.data() + p, scales.data(), scales.size() * 4);
    } else if (sdt == ScaleDtype::F16 && !scales.empty()) {
        auto hs = qpack::fp32_span_to_fp16(scales.data(), scales.size());
        size_t p = out.size();
        out.resize(p + hs.size() * 2);
        std::memcpy(out.data() + p, hs.data(), hs.size() * 2);
    }
    out.insert(out.end(), zps.begin(), zps.end());
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}

uint32_t parse_u32(const std::string& s, const char* name) {
    char* end = nullptr;
    errno = 0;
    unsigned long v = std::strtoul(s.c_str(), &end, 10);
    if (s.empty() || errno != 0 || end != s.c_str() + s.size() || v > 0xFFFFFFFFUL) {
        throw std::runtime_error(std::string("Invalid uint32 for ") + name);
    }
    return static_cast<uint32_t>(v);
}

std::string render_metrics_json(const qpack::ReconstructionMetrics& m) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(8)
       << "{\"mse\": " << m.mse << ", \"rmse\": " << m.rmse << ", \"sqnr_db\": " << m.sqnr_db
       << ", \"cosine_similarity\": " << m.cosine_similarity
       << ", \"max_abs_error\": " << m.max_abs_error << ", \"top_outliers\": [";
    for (size_t i = 0; i < m.top_outliers.size(); ++i) {
        const auto& o = m.top_outliers[i];
        os << (i ? ", " : "") << "{\"index\": " << o.index << ", \"reference\": " << o.reference
           << ", \"reconstructed\": " << o.reconstructed << ", \"abs_error\": " << o.abs_error << "}";
    }
    os << "]}";
    return os.str();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) throw std::runtime_error("Missing subcommand");
        std::string cmd = argv[1], in_p, out_p, ref_p, rep_p, dt_s = "q8_sym", sc_s = "f32";
        uint32_t rows = 0, cols = 0, group_sz = 32, top_k = 5;
        bool raw_f32 = false;
        double min_sqnr = -1e9;

        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::runtime_error("Missing value for " + a);
                return argv[++i];
            };
            if (a == "--input" || a == "-i") in_p = next();
            else if (a == "--output" || a == "-o") out_p = next();
            else if (a == "--reference") ref_p = next();
            else if (a == "--report") rep_p = next();
            else if (a == "--dtype") dt_s = next();
            else if (a == "--scale-dtype") sc_s = next();
            else if (a == "--rows") rows = parse_u32(next(), "--rows");
            else if (a == "--cols") cols = parse_u32(next(), "--cols");
            else if (a == "--group-size") group_sz = parse_u32(next(), "--group-size");
            else if (a == "--top-k") top_k = parse_u32(next(), "--top-k");
            else if (a == "--min-sqnr-db") min_sqnr = std::stod(next());
            else if (a == "--raw-f32") raw_f32 = true;
            else throw std::runtime_error("Unknown option: " + a);
        }
        if (in_p.empty()) throw std::runtime_error("Missing --input");

        if (cmd == "quantize") {
            if (out_p.empty()) throw std::runtime_error("quantize requires --output");
            TensorDtype dt = parse_dtype(dt_s);
            ScaleDtype sdt = sc_s == "f16" ? ScaleDtype::F16
                             : (sc_s == "f32" ? ScaleDtype::F32 : throw std::runtime_error("Bad scale-dtype"));
            auto src = load_source(in_p, rows, cols);
            auto blob = encode_qtns(src.values, src.header.rows, src.header.cols, dt, group_sz, sdt);
            write_exclusive_bytes(out_p, blob.data(), blob.size());

            auto rt = decode_qtns_buffer(blob);
            bool all_fin = true;
            for (size_t i = 0; i < src.values.size(); ++i) {
                if (!std::isfinite(src.values[i]) || !std::isfinite(rt.values[i])) { all_fin = false; break; }
            }
            auto m = all_fin ? qpack::compute_error_metrics(src.values, rt.values, top_k)
                             : qpack::ReconstructionMetrics{};
            std::ostringstream js;
            js << std::fixed << std::setprecision(6) << "{\n  \"dtype\": \"" << dtype_to_name(dt)
               << "\",\n  \"rows\": " << src.header.rows << ",\n  \"cols\": " << src.header.cols
               << ",\n  \"encoded_bytes\": " << blob.size()
               << ",\n  \"compression_ratio\": " << (static_cast<double>(src.values.size() * 4) / blob.size())
               << ",\n  \"metrics\": " << render_metrics_json(m) << "\n}\n";
            if (!rep_p.empty()) write_exclusive_bytes(rep_p, js.str().data(), js.str().size());
            else std::cout << js.str();
        } else if (cmd == "dequantize") {
            if (out_p.empty()) throw std::runtime_error("dequantize requires --output");
            auto dec = decode_qtns_buffer(read_all_bytes(in_p));
            if (raw_f32) write_exclusive_bytes(out_p, dec.values.data(), dec.values.size() * 4);
            else {
                auto blob = encode_qtns(dec.values, dec.header.rows, dec.header.cols,
                                        TensorDtype::F32, 0, ScaleDtype::NONE);
                write_exclusive_bytes(out_p, blob.data(), blob.size());
            }
        } else if (cmd == "inspect") {
            auto dec = decode_qtns_buffer(read_all_bytes(in_p));
            size_t nans = 0, infs = 0, fins = 0;
            double vmin = 0.0, vmax = 0.0, sum = 0.0;
            for (float v : dec.values) {
                if (std::isnan(v)) ++nans;
                else if (std::isinf(v)) ++infs;
                else {
                    vmin = fins == 0 ? v : std::min(vmin, static_cast<double>(v));
                    vmax = fins == 0 ? v : std::max(vmax, static_cast<double>(v));
                    sum += v;
                    ++fins;
                }
            }
            std::ostringstream js;
            js << std::fixed << std::setprecision(6) << "{\n  \"dtype\": \""
               << dtype_to_name(static_cast<TensorDtype>(dec.header.dtype))
               << "\",\n  \"rows\": " << dec.header.rows << ",\n  \"cols\": " << dec.header.cols
               << ",\n  \"group_size\": " << dec.header.group_size << ",\n  \"num_groups\": " << dec.header.num_groups
               << ",\n  \"total_bytes\": " << dec.total_file_bytes << ",\n  \"payload_bytes\": " << dec.header.payload_bytes
               << ",\n  \"nan_count\": " << nans << ",\n  \"inf_count\": " << infs
               << ",\n  \"min\": " << vmin << ",\n  \"max\": " << vmax
               << ",\n  \"mean\": " << (fins ? sum / static_cast<double>(fins) : 0.0) << "\n}\n";
            if (!out_p.empty()) write_exclusive_bytes(out_p, js.str().data(), js.str().size());
            else std::cout << js.str();
        } else if (cmd == "verify") {
            if (ref_p.empty()) throw std::runtime_error("verify requires --reference");
            auto dec = decode_qtns_buffer(read_all_bytes(in_p));
            auto ref = load_source(ref_p, rows ? rows : dec.header.rows, cols ? cols : dec.header.cols);
            if (dec.header.rows != ref.header.rows || dec.header.cols != ref.header.cols) {
                throw std::runtime_error("Shape mismatch in verify");
            }
            auto m = qpack::compute_error_metrics(ref.values, dec.values, top_k);
            std::ostringstream js;
            js << "{\n  \"dtype\": \"" << dtype_to_name(static_cast<TensorDtype>(dec.header.dtype))
               << "\",\n  \"rows\": " << dec.header.rows << ",\n  \"cols\": " << dec.header.cols
               << ",\n  \"metrics\": " << render_metrics_json(m) << "\n}\n";
            if (!out_p.empty()) write_exclusive_bytes(out_p, js.str().data(), js.str().size());
            else std::cout << js.str();
            if (m.sqnr_db < min_sqnr) throw std::runtime_error("SQNR below threshold");
        } else {
            throw std::runtime_error("Unknown subcommand: " + cmd);
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "tensor_quant_cli error: " << ex.what() << "\n";
        return 1;
    }
}
