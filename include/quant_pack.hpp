#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <queue>
#include <stdexcept>
#include <vector>

namespace qpack {

inline uint32_t float_to_bits(float v) noexcept {
    uint32_t u = 0;
    std::memcpy(&u, &v, sizeof(uint32_t));
    return u;
}

inline float bits_to_float(uint32_t u) noexcept {
    float v = 0.0f;
    std::memcpy(&v, &u, sizeof(float));
    return v;
}

inline uint16_t fp32_to_fp16(float value) noexcept {
    uint32_t bits = float_to_bits(value);
    uint32_t sign = (bits >> 16) & 0x8000u;
    uint32_t exp = (bits >> 23) & 0xFFu;
    uint32_t mant = bits & 0x007FFFFFu;

    if (exp == 0xFFu) {
        if (mant == 0u) return static_cast<uint16_t>(sign | 0x7C00u);
        return static_cast<uint16_t>(sign | 0x7E00u | static_cast<uint16_t>(mant >> 13));
    }

    int32_t half_exp = static_cast<int32_t>(exp) - 127 + 15;
    if (half_exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    if (half_exp <= 0) {
        if (half_exp < -10) return static_cast<uint16_t>(sign);
        uint32_t full_mant = 0x00800000u | mant;
        int shift = 14 - half_exp;
        uint32_t half_way = 1u << (shift - 1);
        uint32_t rem = full_mant & ((1u << shift) - 1u);
        uint32_t sub = full_mant >> shift;
        if (rem > half_way || (rem == half_way && (sub & 1u) != 0u)) ++sub;
        return static_cast<uint16_t>(sign | sub);
    }

    uint32_t mant16 = mant >> 13;
    uint32_t rem = mant & 0x1FFFu;
    uint32_t packed = (static_cast<uint32_t>(half_exp) << 10) | mant16;
    if (rem > 0x1000u || (rem == 0x1000u && (mant16 & 1u) != 0u)) ++packed;
    return static_cast<uint16_t>(sign | packed);
}

inline float fp16_to_fp32(uint16_t half_bits) noexcept {
    uint32_t sign = static_cast<uint32_t>(half_bits & 0x8000u) << 16;
    uint32_t exp = (half_bits >> 10) & 0x1Fu;
    uint32_t mant = half_bits & 0x03FFu;

    if (exp == 0u) {
        if (mant == 0u) return bits_to_float(sign);
        uint32_t exp32 = 113u;
        while ((mant & 0x0400u) == 0u) {
            mant <<= 1;
            --exp32;
        }
        return bits_to_float(sign | (exp32 << 23) | ((mant & 0x03FFu) << 13));
    }
    if (exp == 0x1Fu) {
        return bits_to_float(sign | (mant == 0u ? 0x7F800000u : (0x7FC00000u | (mant << 13))));
    }
    return bits_to_float(sign | ((exp + 112u) << 23) | (mant << 13));
}

inline uint16_t fp32_to_bf16(float value) noexcept {
    uint32_t bits = float_to_bits(value);
    if (((bits >> 23) & 0xFFu) == 0xFFu) {
        return static_cast<uint16_t>((bits >> 16) | ((bits & 0x007FFFFFu) ? 0x0040u : 0u));
    }
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>(bits >> 16);
}

inline float bf16_to_fp32(uint16_t bf16_bits) noexcept {
    return bits_to_float(static_cast<uint32_t>(bf16_bits) << 16);
}

inline std::vector<uint16_t> fp32_span_to_fp16(const float* src, size_t n) {
    if (n > 0 && src == nullptr) throw std::invalid_argument("Null fp32 pointer");
    std::vector<uint16_t> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = fp32_to_fp16(src[i]);
    return out;
}

inline std::vector<float> fp16_span_to_fp32(const uint16_t* src, size_t n) {
    if (n > 0 && src == nullptr) throw std::invalid_argument("Null fp16 pointer");
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = fp16_to_fp32(src[i]);
    return out;
}

inline std::vector<uint16_t> fp32_span_to_bf16(const float* src, size_t n) {
    if (n > 0 && src == nullptr) throw std::invalid_argument("Null fp32 pointer");
    std::vector<uint16_t> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = fp32_to_bf16(src[i]);
    return out;
}

inline std::vector<float> bf16_span_to_fp32(const uint16_t* src, size_t n) {
    if (n > 0 && src == nullptr) throw std::invalid_argument("Null bf16 pointer");
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = bf16_to_fp32(src[i]);
    return out;
}

enum class TensorDtype : uint8_t {
    F32 = 0, F16 = 1, BF16 = 2, Q8_SYM = 3, Q8_ASYM = 4, Q4_SYM = 5, Q4_ASYM = 6
};

enum class ScaleDtype : uint8_t { NONE = 0, F32 = 1, F16 = 2 };

struct TensorHeader {
    char magic[4] = {'Q', 'T', 'N', 'S'};
    uint16_t version = 1;
    uint8_t dtype = 0;
    uint8_t scale_dtype = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t group_size = 0;
    uint32_t num_groups = 0;
    uint64_t payload_bytes = 0;
};
static_assert(sizeof(TensorHeader) == 32, "TensorHeader must be 32 bytes");

struct SymQuant8Block {
    uint32_t group_size = 0;
    size_t num_elements = 0;
    std::vector<float> scales;
    std::vector<int8_t> data;
};

struct AsymQuant8Block {
    uint32_t group_size = 0;
    size_t num_elements = 0;
    std::vector<float> scales;
    std::vector<uint8_t> zero_points;
    std::vector<uint8_t> data;
};

struct SymQuant4Block {
    uint32_t group_size = 0;
    size_t num_elements = 0;
    std::vector<float> scales;
    std::vector<uint8_t> packed_data;
};

struct AsymQuant4Block {
    uint32_t group_size = 0;
    size_t num_elements = 0;
    std::vector<float> scales;
    std::vector<uint8_t> zero_points;
    std::vector<uint8_t> packed_data;
};

struct OutlierEntry {
    size_t index = 0;
    float reference = 0.0f;
    float reconstructed = 0.0f;
    float abs_error = 0.0f;
};

struct ReconstructionMetrics {
    size_t count = 0;
    double mse = 0.0;
    double rmse = 0.0;
    double sqnr_db = 0.0;
    double cosine_similarity = 1.0;
    double max_abs_error = 0.0;
    std::vector<OutlierEntry> top_outliers;
};

namespace detail {

inline void check_inputs(const float* src, size_t n, uint32_t group_size) {
    if (group_size == 0) throw std::invalid_argument("group_size must be > 0");
    if (n > 0 && src == nullptr) throw std::invalid_argument("Null tensor pointer");
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(src[i])) throw std::invalid_argument("Non-finite float in quantizer");
    }
}

inline int32_t rne(float x) noexcept { return static_cast<int32_t>(std::nearbyint(x)); }

inline float snap_scale(float s, bool fp16) noexcept {
    return fp16 ? fp16_to_fp32(fp32_to_fp16(s)) : s;
}

}  // namespace detail

inline void pack_nibbles(uint8_t* dst, size_t idx, uint8_t nib) noexcept {
    size_t b = idx >> 1;
    dst[b] = (idx & 1u) == 0u ? static_cast<uint8_t>((dst[b] & 0xF0u) | (nib & 0x0Fu))
                              : static_cast<uint8_t>((dst[b] & 0x0Fu) | ((nib & 0x0Fu) << 4));
}

inline uint8_t unpack_nibble(const uint8_t* src, size_t idx) noexcept {
    uint8_t b = src[idx >> 1];
    return (idx & 1u) == 0u ? static_cast<uint8_t>(b & 0x0Fu) : static_cast<uint8_t>((b >> 4) & 0x0Fu);
}

inline SymQuant8Block quantize_q8_symmetric(const float* src,
                                            size_t n,
                                            uint32_t group_size,
                                            bool fp16_scales = false) {
    detail::check_inputs(src, n, group_size);
    size_t ng = (n + group_size - 1) / group_size;
    SymQuant8Block out{group_size, n, std::vector<float>(ng, 0.0f), std::vector<int8_t>(n, 0)};
    for (size_t g = 0; g < ng; ++g) {
        size_t b = g * group_size, e = std::min(n, b + group_size);
        float amax = 0.0f;
        for (size_t i = b; i < e; ++i) amax = std::max(amax, std::fabs(src[i]));
        float s = detail::snap_scale(amax / 127.0f, fp16_scales);
        if (s <= 0.0f || !std::isfinite(s)) continue;
        out.scales[g] = s;
        float inv = 1.0f / s;
        for (size_t i = b; i < e; ++i) {
            out.data[i] = static_cast<int8_t>(std::clamp(detail::rne(src[i] * inv), -127, 127));
        }
    }
    return out;
}

inline std::vector<float> dequantize_q8_symmetric(const SymQuant8Block& blk) {
    if (blk.group_size == 0) throw std::invalid_argument("group_size == 0");
    size_t ng = (blk.num_elements + blk.group_size - 1) / blk.group_size;
    if (blk.scales.size() != ng || blk.data.size() != blk.num_elements) {
        throw std::invalid_argument("Malformed SymQuant8Block");
    }
    std::vector<float> dst(blk.num_elements, 0.0f);
    for (size_t g = 0; g < ng; ++g) {
        size_t b = g * blk.group_size, e = std::min(blk.num_elements, b + blk.group_size);
        for (size_t i = b; i < e; ++i) dst[i] = static_cast<float>(blk.data[i]) * blk.scales[g];
    }
    return dst;
}

inline AsymQuant8Block quantize_q8_asymmetric(const float* src,
                                              size_t n,
                                              uint32_t group_size,
                                              bool fp16_scales = false) {
    detail::check_inputs(src, n, group_size);
    size_t ng = (n + group_size - 1) / group_size;
    AsymQuant8Block out{group_size, n, std::vector<float>(ng, 0.0f),
                        std::vector<uint8_t>(ng, 0u), std::vector<uint8_t>(n, 0u)};
    for (size_t g = 0; g < ng; ++g) {
        size_t b = g * group_size, e = std::min(n, b + group_size);
        float lo = 0.0f, hi = 0.0f;
        for (size_t i = b; i < e; ++i) {
            lo = std::min(lo, src[i]);
            hi = std::max(hi, src[i]);
        }
        float s = detail::snap_scale((hi - lo) / 255.0f, fp16_scales);
        if (s <= 0.0f || !std::isfinite(s)) continue;
        int32_t zp = std::clamp(detail::rne(-lo / s), 0, 255);
        out.scales[g] = s;
        out.zero_points[g] = static_cast<uint8_t>(zp);
        float inv = 1.0f / s;
        for (size_t i = b; i < e; ++i) {
            out.data[i] = static_cast<uint8_t>(std::clamp(detail::rne(src[i] * inv) + zp, 0, 255));
        }
    }
    return out;
}

inline std::vector<float> dequantize_q8_asymmetric(const AsymQuant8Block& blk) {
    if (blk.group_size == 0) throw std::invalid_argument("group_size == 0");
    size_t ng = (blk.num_elements + blk.group_size - 1) / blk.group_size;
    if (blk.scales.size() != ng || blk.zero_points.size() != ng || blk.data.size() != blk.num_elements) {
        throw std::invalid_argument("Malformed AsymQuant8Block");
    }
    std::vector<float> dst(blk.num_elements, 0.0f);
    for (size_t g = 0; g < ng; ++g) {
        size_t b = g * blk.group_size, e = std::min(blk.num_elements, b + blk.group_size);
        int32_t zp = static_cast<int32_t>(blk.zero_points[g]);
        for (size_t i = b; i < e; ++i) {
            dst[i] = static_cast<float>(static_cast<int32_t>(blk.data[i]) - zp) * blk.scales[g];
        }
    }
    return dst;
}

inline SymQuant4Block quantize_q4_symmetric(const float* src,
                                            size_t n,
                                            uint32_t group_size,
                                            bool fp16_scales = false) {
    detail::check_inputs(src, n, group_size);
    size_t ng = (n + group_size - 1) / group_size;
    SymQuant4Block out{group_size, n, std::vector<float>(ng, 0.0f), std::vector<uint8_t>((n + 1) / 2, 0x88u)};
    for (size_t g = 0; g < ng; ++g) {
        size_t b = g * group_size, e = std::min(n, b + group_size);
        float amax = 0.0f;
        for (size_t i = b; i < e; ++i) amax = std::max(amax, std::fabs(src[i]));
        float s = detail::snap_scale(amax / 7.0f, fp16_scales);
        if (s <= 0.0f || !std::isfinite(s)) continue;
        out.scales[g] = s;
        float inv = 1.0f / s;
        for (size_t i = b; i < e; ++i) {
            int32_t q = std::clamp(detail::rne(src[i] * inv), -7, 7);
            pack_nibbles(out.packed_data.data(), i, static_cast<uint8_t>(q + 8));
        }
    }
    return out;
}

inline std::vector<float> dequantize_q4_symmetric(const SymQuant4Block& blk) {
    if (blk.group_size == 0) throw std::invalid_argument("group_size == 0");
    size_t ng = (blk.num_elements + blk.group_size - 1) / blk.group_size;
    if (blk.scales.size() != ng || blk.packed_data.size() != (blk.num_elements + 1) / 2) {
        throw std::invalid_argument("Malformed SymQuant4Block");
    }
    std::vector<float> dst(blk.num_elements, 0.0f);
    for (size_t g = 0; g < ng; ++g) {
        size_t b = g * blk.group_size, e = std::min(blk.num_elements, b + blk.group_size);
        for (size_t i = b; i < e; ++i) {
            int32_t q = static_cast<int32_t>(unpack_nibble(blk.packed_data.data(), i)) - 8;
            dst[i] = static_cast<float>(q) * blk.scales[g];
        }
    }
    return dst;
}

inline AsymQuant4Block quantize_q4_asymmetric(const float* src,
                                              size_t n,
                                              uint32_t group_size,
                                              bool fp16_scales = false) {
    detail::check_inputs(src, n, group_size);
    size_t ng = (n + group_size - 1) / group_size;
    AsymQuant4Block out{group_size, n, std::vector<float>(ng, 0.0f),
                        std::vector<uint8_t>(ng, 0u), std::vector<uint8_t>((n + 1) / 2, 0u)};
    for (size_t g = 0; g < ng; ++g) {
        size_t b = g * group_size, e = std::min(n, b + group_size);
        float lo = 0.0f, hi = 0.0f;
        for (size_t i = b; i < e; ++i) {
            lo = std::min(lo, src[i]);
            hi = std::max(hi, src[i]);
        }
        float s = detail::snap_scale((hi - lo) / 15.0f, fp16_scales);
        if (s <= 0.0f || !std::isfinite(s)) continue;
        int32_t zp = std::clamp(detail::rne(-lo / s), 0, 15);
        out.scales[g] = s;
        out.zero_points[g] = static_cast<uint8_t>(zp);
        float inv = 1.0f / s;
        for (size_t i = b; i < e; ++i) {
            int32_t q = std::clamp(detail::rne(src[i] * inv) + zp, 0, 15);
            pack_nibbles(out.packed_data.data(), i, static_cast<uint8_t>(q));
        }
    }
    return out;
}

inline std::vector<float> dequantize_q4_asymmetric(const AsymQuant4Block& blk) {
    if (blk.group_size == 0) throw std::invalid_argument("group_size == 0");
    size_t ng = (blk.num_elements + blk.group_size - 1) / blk.group_size;
    if (blk.scales.size() != ng || blk.zero_points.size() != ng ||
        blk.packed_data.size() != (blk.num_elements + 1) / 2) {
        throw std::invalid_argument("Malformed AsymQuant4Block");
    }
    std::vector<float> dst(blk.num_elements, 0.0f);
    for (size_t g = 0; g < ng; ++g) {
        size_t b = g * blk.group_size, e = std::min(blk.num_elements, b + blk.group_size);
        int32_t zp = static_cast<int32_t>(blk.zero_points[g]);
        for (size_t i = b; i < e; ++i) {
            int32_t q = static_cast<int32_t>(unpack_nibble(blk.packed_data.data(), i));
            dst[i] = static_cast<float>(q - zp) * blk.scales[g];
        }
    }
    return dst;
}

inline ReconstructionMetrics compute_error_metrics(const std::vector<float>& ref,
                                                   const std::vector<float>& recon,
                                                   size_t top_k = 5) {
    if (ref.size() != recon.size()) throw std::invalid_argument("Tensor size mismatch");
    ReconstructionMetrics m;
    m.count = ref.size();
    if (m.count == 0) return m;

    auto cmp = [](const OutlierEntry& a, const OutlierEntry& b) {
        return a.abs_error != b.abs_error ? (a.abs_error > b.abs_error) : (a.index < b.index);
    };
    std::priority_queue<OutlierEntry, std::vector<OutlierEntry>, decltype(cmp)> heap(cmp);

    long double sig_p = 0.0L, err_p = 0.0L, dot = 0.0L, rec_p = 0.0L;
    for (size_t i = 0; i < m.count; ++i) {
        long double r = ref[i], c = recon[i];
        if (!std::isfinite(r) || !std::isfinite(c)) throw std::invalid_argument("Non-finite metric input");
        long double d = r - c;
        double ae = static_cast<double>(std::fabs(d));
        sig_p += r * r;
        err_p += d * d;
        dot += r * c;
        rec_p += c * c;
        m.max_abs_error = std::max(m.max_abs_error, ae);
        if (top_k > 0) {
            OutlierEntry cur{i, ref[i], recon[i], static_cast<float>(ae)};
            if (heap.size() < top_k) heap.push(cur);
            else if (cmp(cur, heap.top())) { heap.pop(); heap.push(cur); }
        }
    }

    m.mse = static_cast<double>(err_p / static_cast<long double>(m.count));
    m.rmse = std::sqrt(m.mse);
    m.sqnr_db = (err_p == 0.0L) ? (sig_p > 0.0L ? 120.0 : 0.0)
                                : (sig_p == 0.0L ? 0.0 : static_cast<double>(10.0L * std::log10(sig_p / err_p)));
    long double denom = std::sqrt(sig_p) * std::sqrt(rec_p);
    m.cosine_similarity = (denom == 0.0L) ? ((sig_p == 0.0L && rec_p == 0.0L) ? 1.0 : 0.0)
                                          : std::clamp(static_cast<double>(dot / denom), -1.0, 1.0);
    while (!heap.empty()) {
        m.top_outliers.push_back(heap.top());
        heap.pop();
    }
    std::reverse(m.top_outliers.begin(), m.top_outliers.end());
    return m;
}

}  // namespace qpack
