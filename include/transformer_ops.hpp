#pragma once

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace rt_ops {

constexpr uint32_t kEmb1HeaderBytes = 32;
constexpr uint32_t kEmb1Version = 1;
constexpr uint32_t kEmb1FlagRowSymInt8 = 1;

constexpr uint32_t kLmh1HeaderBytes = 64;
constexpr uint32_t kLmh1Version = 1;
constexpr uint32_t kLmh1FlagHasNorm = 1;

inline uint32_t read_u32_le(const uint8_t* p) noexcept {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

inline void write_u32_le(uint8_t* p, uint32_t v) noexcept {
  p[0] = static_cast<uint8_t>(v & 0xffu);
  p[1] = static_cast<uint8_t>((v >> 8) & 0xffu);
  p[2] = static_cast<uint8_t>((v >> 16) & 0xffu);
  p[3] = static_cast<uint8_t>((v >> 24) & 0xffu);
}

inline float bits_to_f32(uint32_t bits) noexcept {
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(float));
  return out;
}

inline uint32_t f32_to_bits(float v) noexcept {
  uint32_t bits = 0;
  std::memcpy(&bits, &v, sizeof(uint32_t));
  return bits;
}

inline float read_f32_le(const uint8_t* p) noexcept { return bits_to_f32(read_u32_le(p)); }
inline void write_f32_le(uint8_t* p, float v) noexcept { write_u32_le(p, f32_to_bits(v)); }
inline float bf16_bits_to_f32(uint16_t bf16) noexcept {
  return bits_to_f32(static_cast<uint32_t>(bf16) << 16);
}

inline size_t align_up(size_t value, size_t alignment) noexcept {
  if (alignment <= 1) return value;
  const size_t rem = value % alignment;
  return rem == 0 ? value : (value + (alignment - rem));
}

inline bool pread_exact(int fd, void* dst, size_t nbytes, off_t offset) noexcept {
  auto* ptr = static_cast<uint8_t*>(dst);
  size_t done = 0;
  while (done < nbytes) {
    const ssize_t r = ::pread(fd, ptr + done, nbytes - done, offset + static_cast<off_t>(done));
    if (r > 0) {
      done += static_cast<size_t>(r);
    } else if (r < 0 && errno == EINTR) {
      continue;
    } else {
      return false;
    }
  }
  return true;
}

struct QuantAuditStats {
  double cos_min = 1.0;
  double cos_mean = 1.0;
  double max_abs_err = 0.0;
};

inline void quantize_row_sym_i8(
    const float* src, uint32_t cols, int8_t* dst_codes, float* dst_scale) noexcept {
  float abs_max = 0.0f;
  for (uint32_t d = 0; d < cols; ++d) abs_max = std::max(abs_max, std::fabs(src[d]));
  const float scale = std::max(abs_max / 127.0f, 1e-12f);
  *dst_scale = scale;
  for (uint32_t d = 0; d < cols; ++d) {
    const float clamped = std::max(-127.0f, std::min(127.0f, std::nearbyintf(src[d] / scale)));
    dst_codes[d] = static_cast<int8_t>(static_cast<int32_t>(clamped));
  }
}

inline QuantAuditStats compute_row_quant_audit(
    const float* orig, const int8_t* codes, const float* scales, uint32_t rows, uint32_t cols) noexcept {
  QuantAuditStats st{};
  double cos_sum = 0.0, cos_min = 1.0, max_err = 0.0;
  uint32_t valid_rows = 0;
  for (uint32_t r = 0; r < rows; ++r) {
    const float* row_orig = orig + static_cast<size_t>(r) * cols;
    const int8_t* row_q = codes + static_cast<size_t>(r) * cols;
    const float s = scales[r];
    double dot = 0.0, norm_orig = 0.0, norm_deq = 0.0;
    for (uint32_t d = 0; d < cols; ++d) {
      const double o64 = static_cast<double>(row_orig[d]);
      const double q64 = static_cast<double>(static_cast<float>(row_q[d]) * s);
      max_err = std::max(max_err, std::fabs(o64 - q64));
      dot += o64 * q64;
      norm_orig += o64 * o64;
      norm_deq += q64 * q64;
    }
    const double denom = std::sqrt(norm_orig) * std::sqrt(norm_deq);
    if (denom > 0.0) {
      const double c = std::max(-1.0, std::min(1.0, dot / denom));
      if (valid_rows == 0 || c < cos_min) cos_min = c;
      cos_sum += c;
      ++valid_rows;
    }
  }
  st.max_abs_err = max_err;
  st.cos_min = valid_rows > 0 ? cos_min : 1.0;
  st.cos_mean = valid_rows > 0 ? (cos_sum / static_cast<double>(valid_rows)) : 1.0;
  return st;
}

inline bool write_emb1_file(
    const std::string& path, const float* weights, uint32_t vocab_size, uint32_t hidden_size,
    QuantAuditStats* stats_out = nullptr) {
  if (!weights || vocab_size == 0 || hidden_size == 0) return false;
  const uint32_t row_stride = 4u + hidden_size;
  uint8_t hdr[kEmb1HeaderBytes];
  std::memset(hdr, 0, sizeof(hdr));
  std::memcpy(hdr, "EMB1", 4);
  write_u32_le(hdr + 4, kEmb1Version);
  write_u32_le(hdr + 8, vocab_size);
  write_u32_le(hdr + 12, hidden_size);
  write_u32_le(hdr + 16, row_stride);
  write_u32_le(hdr + 20, kEmb1FlagRowSymInt8);

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out.is_open()) return false;
  out.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
  if (!out.good()) return false;

  std::vector<int8_t> all_codes;
  std::vector<float> all_scales;
  if (stats_out) {
    all_codes.resize(static_cast<size_t>(vocab_size) * hidden_size);
    all_scales.resize(vocab_size);
  }
  std::vector<uint8_t> row_buf(row_stride);
  std::vector<int8_t> row_codes(hidden_size);
  for (uint32_t r = 0; r < vocab_size; ++r) {
    float scale = 0.0f;
    const float* src_row = weights + static_cast<size_t>(r) * hidden_size;
    int8_t* code_dst = stats_out ? (all_codes.data() + static_cast<size_t>(r) * hidden_size) : row_codes.data();
    quantize_row_sym_i8(src_row, hidden_size, code_dst, &scale);
    if (stats_out) all_scales[r] = scale;
    write_f32_le(row_buf.data(), scale);
    std::memcpy(row_buf.data() + 4, code_dst, hidden_size);
    out.write(reinterpret_cast<const char*>(row_buf.data()), static_cast<std::streamsize>(row_stride));
    if (!out.good()) return false;
  }
  out.flush();
  if (!out.good()) return false;
  if (stats_out) {
    *stats_out = compute_row_quant_audit(weights, all_codes.data(), all_scales.data(), vocab_size, hidden_size);
  }
  return true;
}

class EmbI8Reader {
 public:
  EmbI8Reader() = default;
  EmbI8Reader(const EmbI8Reader&) = delete;
  EmbI8Reader& operator=(const EmbI8Reader&) = delete;
  EmbI8Reader(EmbI8Reader&& other) noexcept { *this = std::move(other); }
  EmbI8Reader& operator=(EmbI8Reader&& other) noexcept {
    if (this != &other) {
      close();
      fd_ = other.fd_;
      vocab_size_ = other.vocab_size_;
      hidden_size_ = other.hidden_size_;
      row_stride_ = other.row_stride_;
      flags_ = other.flags_;
      other.fd_ = -1;
      other.vocab_size_ = other.hidden_size_ = other.row_stride_ = other.flags_ = 0;
    }
    return *this;
  }
  ~EmbI8Reader() { close(); }

  int open(const std::string& path, uint32_t expected_hidden = 0) noexcept {
    close();
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return -1;
    uint8_t hdr[kEmb1HeaderBytes];
    if (!pread_exact(fd, hdr, sizeof(hdr), 0)) {
      ::close(fd);
      return -2;
    }
    const uint32_t version = read_u32_le(hdr + 4);
    const uint32_t vocab = read_u32_le(hdr + 8);
    const uint32_t hidden = read_u32_le(hdr + 12);
    const uint32_t stride = read_u32_le(hdr + 16);
    const uint32_t flags = read_u32_le(hdr + 20);
    if (std::memcmp(hdr, "EMB1", 4) != 0 || version != kEmb1Version || vocab == 0 || hidden == 0 ||
        (expected_hidden != 0 && hidden != expected_hidden) || stride != 4u + hidden ||
        flags != kEmb1FlagRowSymInt8) {
      ::close(fd);
      return -3;
    }
    struct stat st{};
    if (::fstat(fd, &st) != 0 || st.st_size < 0) {
      ::close(fd);
      return -4;
    }
    const uint64_t min_bytes =
        static_cast<uint64_t>(kEmb1HeaderBytes) + static_cast<uint64_t>(vocab) * stride;
    if (static_cast<uint64_t>(st.st_size) < min_bytes) {
      ::close(fd);
      return -5;
    }
#ifdef POSIX_FADV_RANDOM
    const int adv_rc = ::posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
    (void)adv_rc;
#endif
    fd_ = fd;
    vocab_size_ = vocab;
    hidden_size_ = hidden;
    row_stride_ = stride;
    flags_ = flags;
    return 0;
  }

  void close() noexcept {
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
    vocab_size_ = hidden_size_ = row_stride_ = flags_ = 0;
  }

  bool is_open() const noexcept { return fd_ >= 0; }
  uint32_t vocab_size() const noexcept { return vocab_size_; }
  uint32_t hidden_size() const noexcept { return hidden_size_; }
  uint32_t row_stride() const noexcept { return row_stride_; }
  uint32_t flags() const noexcept { return flags_; }

  int lookup_f32(int32_t token_id, float* dst) const {
    if (fd_ < 0 || !dst || token_id < 0 || static_cast<uint32_t>(token_id) >= vocab_size_) return -1;
    std::vector<uint8_t> buf(row_stride_);
    const off_t off = static_cast<off_t>(kEmb1HeaderBytes) +
                      static_cast<off_t>(token_id) * static_cast<off_t>(row_stride_);
    if (!pread_exact(fd_, buf.data(), row_stride_, off)) return -2;
    const float scale = read_f32_le(buf.data());
    const auto* codes = reinterpret_cast<const int8_t*>(buf.data() + 4);
    for (uint32_t d = 0; d < hidden_size_; ++d) dst[d] = static_cast<float>(codes[d]) * scale;
    return 0;
  }

  static int lookup_bf16_raw_file(
      const std::string& path, uint32_t hidden_size, int32_t token_id, float* dst) {
    if (hidden_size == 0 || token_id < 0 || !dst) return -1;
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return -2;
    struct stat st{};
    if (::fstat(fd, &st) != 0 || st.st_size < 0) {
      ::close(fd);
      return -3;
    }
    const uint64_t row_bytes = static_cast<uint64_t>(hidden_size) * 2u;
    if (static_cast<uint64_t>(st.st_size) < (static_cast<uint64_t>(token_id) + 1u) * row_bytes) {
      ::close(fd);
      return -4;
    }
    std::vector<uint8_t> raw(static_cast<size_t>(row_bytes));
    const bool ok = pread_exact(
        fd, raw.data(), raw.size(), static_cast<off_t>(static_cast<uint64_t>(token_id) * row_bytes));
    ::close(fd);
    if (!ok) return -5;
    for (uint32_t d = 0; d < hidden_size; ++d) {
      const uint16_t u16 = static_cast<uint16_t>(raw[2u * d]) |
                           static_cast<uint16_t>(static_cast<uint16_t>(raw[2u * d + 1u]) << 8);
      dst[d] = bf16_bits_to_f32(u16);
    }
    return 0;
  }

 private:
  int fd_ = -1;
  uint32_t vocab_size_ = 0;
  uint32_t hidden_size_ = 0;
  uint32_t row_stride_ = 0;
  uint32_t flags_ = 0;
};

inline void rms_norm_f32(
    const float* x, const float* weight, uint32_t dim, float eps, float* out) noexcept {
  double sum_sq = 0.0;
  for (uint32_t i = 0; i < dim; ++i) sum_sq += static_cast<double>(x[i]) * static_cast<double>(x[i]);
  const float inv_rms = static_cast<float>(
      1.0 / std::sqrt(sum_sq / static_cast<double>(dim) + static_cast<double>(eps)));
  for (uint32_t i = 0; i < dim; ++i) {
    out[i] = weight ? (x[i] * inv_rms * weight[i]) : (x[i] * inv_rms);
  }
}

class WorkerPool {
 public:
  explicit WorkerPool(uint32_t num_threads = 1) : n_threads_(std::max<uint32_t>(1u, num_threads)) {
    for (uint32_t t = 1; t < n_threads_; ++t) {
      workers_.emplace_back([this, t]() { worker_loop(t); });
    }
  }
  WorkerPool(const WorkerPool&) = delete;
  WorkerPool& operator=(const WorkerPool&) = delete;

  ~WorkerPool() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
    }
    cv_go_.notify_all();
    for (auto& th : workers_) {
      if (th.joinable()) th.join();
    }
  }

  uint32_t num_threads() const noexcept { return n_threads_; }

  void parallel_stripes(uint32_t total_rows, const std::function<void(uint32_t, uint32_t, uint32_t)>& fn) {
    if (total_rows == 0) return;
    if (n_threads_ == 1u || total_rows == 1u) {
      fn(0u, 0u, total_rows);
      return;
    }
    const uint32_t per = (total_rows + n_threads_ - 1u) / n_threads_;
    {
      std::lock_guard<std::mutex> lk(mu_);
      task_fn_ = &fn;
      total_rows_ = total_rows;
      per_thread_ = per;
      remaining_ = n_threads_ - 1u;
      ++generation_;
    }
    cv_go_.notify_all();
    fn(0u, 0u, std::min(total_rows, per));
    std::unique_lock<std::mutex> lk(mu_);
    cv_done_.wait(lk, [this]() { return remaining_ == 0u; });
    task_fn_ = nullptr;
  }

 private:
  void worker_loop(uint32_t tid) {
    uint64_t seen_gen = 0;
    for (;;) {
      const std::function<void(uint32_t, uint32_t, uint32_t)>* fn = nullptr;
      uint32_t total = 0, per = 0;
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_go_.wait(lk, [this, seen_gen]() { return stop_ || generation_ != seen_gen; });
        if (stop_) return;
        seen_gen = generation_;
        fn = task_fn_;
        total = total_rows_;
        per = per_thread_;
      }
      const uint32_t r0 = std::min(total, tid * per);
      const uint32_t r1 = std::min(total, r0 + per);
      if (fn && r0 < r1) (*fn)(tid, r0, r1);
      std::lock_guard<std::mutex> lk(mu_);
      if (--remaining_ == 0u) cv_done_.notify_one();
    }
  }

  uint32_t n_threads_ = 1;
  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable cv_go_, cv_done_;
  uint64_t generation_ = 0;
  uint32_t remaining_ = 0, total_rows_ = 0, per_thread_ = 0;
  bool stop_ = false;
  const std::function<void(uint32_t, uint32_t, uint32_t)>* task_fn_ = nullptr;
};

inline bool write_lmh1_file(
    const std::string& path, const float* weights, uint32_t vocab_size, uint32_t hidden_size,
    const float* norm_weight = nullptr, float rms_eps = 1e-6f, QuantAuditStats* stats_out = nullptr) {
  if (!weights || vocab_size == 0 || hidden_size == 0) return false;
  const size_t scale_off = kLmh1HeaderBytes;
  const size_t scale_bytes = static_cast<size_t>(vocab_size) * sizeof(float);
  const size_t codes_off = align_up(scale_off + scale_bytes, 64u);
  const size_t codes_bytes = static_cast<size_t>(vocab_size) * hidden_size;
  const size_t norm_off = norm_weight ? align_up(codes_off + codes_bytes, 64u) : 0u;
  const uint32_t flags = norm_weight ? kLmh1FlagHasNorm : 0u;

  std::vector<int8_t> codes(codes_bytes);
  std::vector<float> scales(vocab_size);
  for (uint32_t r = 0; r < vocab_size; ++r) {
    quantize_row_sym_i8(
        weights + static_cast<size_t>(r) * hidden_size, hidden_size,
        codes.data() + static_cast<size_t>(r) * hidden_size, &scales[r]);
  }
  if (stats_out) *stats_out = compute_row_quant_audit(weights, codes.data(), scales.data(), vocab_size, hidden_size);

  uint8_t hdr[kLmh1HeaderBytes];
  std::memset(hdr, 0, sizeof(hdr));
  std::memcpy(hdr, "LMH1", 4);
  write_u32_le(hdr + 4, kLmh1Version);
  write_u32_le(hdr + 8, vocab_size);
  write_u32_le(hdr + 12, hidden_size);
  write_u32_le(hdr + 16, flags);
  write_u32_le(hdr + 20, static_cast<uint32_t>(scale_off));
  write_u32_le(hdr + 24, static_cast<uint32_t>(codes_off));
  write_u32_le(hdr + 28, static_cast<uint32_t>(norm_off));
  write_u32_le(hdr + 32, f32_to_bits(rms_eps));

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out.is_open()) return false;
  out.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
  std::vector<uint8_t> scale_le(scale_bytes);
  for (uint32_t r = 0; r < vocab_size; ++r) write_f32_le(scale_le.data() + 4u * r, scales[r]);
  out.write(reinterpret_cast<const char*>(scale_le.data()), static_cast<std::streamsize>(scale_bytes));
  if (codes_off > scale_off + scale_bytes) {
    std::vector<char> zeros(codes_off - (scale_off + scale_bytes), 0);
    out.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
  }
  out.write(reinterpret_cast<const char*>(codes.data()), static_cast<std::streamsize>(codes_bytes));
  if (norm_weight) {
    if (norm_off > codes_off + codes_bytes) {
      std::vector<char> zeros(norm_off - (codes_off + codes_bytes), 0);
      out.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
    }
    std::vector<uint8_t> norm_le(static_cast<size_t>(hidden_size) * 4u);
    for (uint32_t d = 0; d < hidden_size; ++d) write_f32_le(norm_le.data() + 4u * d, norm_weight[d]);
    out.write(reinterpret_cast<const char*>(norm_le.data()), static_cast<std::streamsize>(norm_le.size()));
  }
  out.flush();
  return out.good();
}

class QuantLmHead {
 public:
  int load(const std::string& path, const std::string& external_norm_path = "") {
    codes_.clear();
    scales_.clear();
    norm_weight_.clear();
    vocab_size_ = hidden_size_ = 0;
    rms_eps_ = 1e-6f;

    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return -1;
    struct stat st{};
    if (::fstat(fd, &st) != 0 || st.st_size < static_cast<off_t>(kLmh1HeaderBytes)) {
      ::close(fd);
      return -2;
    }
    const uint64_t file_size = static_cast<uint64_t>(st.st_size);
    uint8_t hdr[kLmh1HeaderBytes];
    if (!pread_exact(fd, hdr, sizeof(hdr), 0)) {
      ::close(fd);
      return -3;
    }
    const uint32_t version = read_u32_le(hdr + 4), vocab = read_u32_le(hdr + 8);
    const uint32_t hidden = read_u32_le(hdr + 12), flags = read_u32_le(hdr + 16);
    const uint32_t scale_off = read_u32_le(hdr + 20), codes_off = read_u32_le(hdr + 24);
    const uint32_t norm_off = read_u32_le(hdr + 28), eps_bits = read_u32_le(hdr + 32);

    if (std::memcmp(hdr, "LMH1", 4) != 0 || version != kLmh1Version || vocab == 0 || hidden == 0) {
      ::close(fd);
      return -4;
    }
    const uint64_t scale_bytes = static_cast<uint64_t>(vocab) * 4u;
    const uint64_t codes_bytes = static_cast<uint64_t>(vocab) * hidden;
    if (scale_off < kLmh1HeaderBytes || scale_off + scale_bytes > codes_off || codes_off + codes_bytes > file_size) {
      ::close(fd);
      return -5;
    }
    const bool has_norm = (flags & kLmh1FlagHasNorm) != 0u;
    const uint64_t norm_bytes = static_cast<uint64_t>(hidden) * 4u;
    if (has_norm && (norm_off < codes_off + codes_bytes || norm_off + norm_bytes > file_size)) {
      ::close(fd);
      return -6;
    }

    std::vector<uint8_t> raw_scales(static_cast<size_t>(scale_bytes));
    codes_.resize(static_cast<size_t>(codes_bytes));
    if (!pread_exact(fd, raw_scales.data(), raw_scales.size(), static_cast<off_t>(scale_off)) ||
        !pread_exact(fd, codes_.data(), codes_.size(), static_cast<off_t>(codes_off))) {
      ::close(fd);
      return -7;
    }
    scales_.resize(vocab);
    for (uint32_t r = 0; r < vocab; ++r) scales_[r] = read_f32_le(raw_scales.data() + 4u * r);
    if (has_norm) {
      std::vector<uint8_t> raw_norm(static_cast<size_t>(norm_bytes));
      if (!pread_exact(fd, raw_norm.data(), raw_norm.size(), static_cast<off_t>(norm_off))) {
        ::close(fd);
        return -8;
      }
      norm_weight_.resize(hidden);
      for (uint32_t d = 0; d < hidden; ++d) norm_weight_[d] = read_f32_le(raw_norm.data() + 4u * d);
    }
    ::close(fd);

    if (!external_norm_path.empty()) {
      std::ifstream nf(external_norm_path, std::ios::binary);
      if (!nf.is_open()) return -9;
      std::vector<uint8_t> raw_norm(static_cast<size_t>(hidden) * 4u);
      nf.read(reinterpret_cast<char*>(raw_norm.data()), static_cast<std::streamsize>(raw_norm.size()));
      if (nf.gcount() != static_cast<std::streamsize>(raw_norm.size())) return -10;
      norm_weight_.resize(hidden);
      for (uint32_t d = 0; d < hidden; ++d) norm_weight_[d] = read_f32_le(raw_norm.data() + 4u * d);
    }

    vocab_size_ = vocab;
    hidden_size_ = hidden;
    const float parsed_eps = bits_to_f32(eps_bits);
    rms_eps_ = (parsed_eps > 0.0f && std::isfinite(parsed_eps)) ? parsed_eps : 1e-6f;
    return 0;
  }

  uint32_t vocab_size() const noexcept { return vocab_size_; }
  uint32_t hidden_size() const noexcept { return hidden_size_; }
  float rms_eps() const noexcept { return rms_eps_; }
  bool has_norm_weight() const noexcept { return !norm_weight_.empty(); }

  int32_t forward(
      WorkerPool& pool, const float* hidden_in, float* logits_out, bool apply_norm = true) const {
    if (vocab_size_ == 0 || hidden_size_ == 0 || !hidden_in) return -1;
    std::vector<float> normed;
    const float* x = hidden_in;
    if (apply_norm && !norm_weight_.empty()) {
      normed.resize(hidden_size_);
      rms_norm_f32(hidden_in, norm_weight_.data(), hidden_size_, rms_eps_, normed.data());
      x = normed.data();
    }
    const uint32_t nt = pool.num_threads();
    std::vector<float> stripe_best_val(nt, -std::numeric_limits<float>::infinity());
    std::vector<int32_t> stripe_best_idx(nt, -1);

    pool.parallel_stripes(vocab_size_, [&](uint32_t tid, uint32_t r0, uint32_t r1) {
      float best_v = -std::numeric_limits<float>::infinity();
      int32_t best_i = -1;
      for (uint32_t r = r0; r < r1; ++r) {
        const int8_t* row_codes = codes_.data() + static_cast<size_t>(r) * hidden_size_;
        double acc = 0.0;
        for (uint32_t d = 0; d < hidden_size_; ++d) {
          acc += static_cast<double>(row_codes[d]) * static_cast<double>(x[d]);
        }
        const float val = static_cast<float>(acc * static_cast<double>(scales_[r]));
        if (logits_out) logits_out[r] = val;
        if (val > best_v) {
          best_v = val;
          best_i = static_cast<int32_t>(r);
        }
      }
      stripe_best_val[tid] = best_v;
      stripe_best_idx[tid] = best_i;
    });

    float global_best = -std::numeric_limits<float>::infinity();
    int32_t global_idx = -1;
    for (uint32_t t = 0; t < nt; ++t) {
      if (stripe_best_idx[t] >= 0 && stripe_best_val[t] > global_best) {
        global_best = stripe_best_val[t];
        global_idx = stripe_best_idx[t];
      }
    }
    return global_idx;
  }

 private:
  uint32_t vocab_size_ = 0;
  uint32_t hidden_size_ = 0;
  float rms_eps_ = 1e-6f;
  std::vector<int8_t> codes_;
  std::vector<float> scales_;
  std::vector<float> norm_weight_;
};

struct SamplerConfig {
  float temperature = 1.0f;
  uint32_t top_k = 0;
  float top_p = 1.0f;
};

struct TokenProb {
  int32_t token_id = 0;
  double prob = 0.0;
};

inline std::vector<TokenProb> build_sampling_distribution(
    const float* logits, uint32_t vocab_size, const SamplerConfig& cfg) {
  std::vector<TokenProb> result;
  if (!logits || vocab_size == 0) return result;
  if (cfg.temperature <= 0.0f || cfg.top_k == 1u) {
    float best_v = -std::numeric_limits<float>::infinity();
    int32_t best_i = 0;
    for (uint32_t i = 0; i < vocab_size; ++i) {
      if (logits[i] > best_v) {
        best_v = logits[i];
        best_i = static_cast<int32_t>(i);
      }
    }
    result.push_back(TokenProb{best_i, 1.0});
    return result;
  }

  std::vector<std::pair<float, int32_t>> order(vocab_size);
  for (uint32_t i = 0; i < vocab_size; ++i) order[i] = {logits[i], static_cast<int32_t>(i)};
  const uint32_t k_keep = (cfg.top_k > 0u && cfg.top_k < vocab_size) ? cfg.top_k : vocab_size;
  auto cmp = [](const std::pair<float, int32_t>& a, const std::pair<float, int32_t>& b) {
    return (a.first != b.first) ? (a.first > b.first) : (a.second < b.second);
  };
  if (k_keep < vocab_size) {
    std::partial_sort(order.begin(), order.begin() + k_keep, order.end(), cmp);
    order.resize(k_keep);
  } else {
    std::sort(order.begin(), order.end(), cmp);
  }

  const double inv_temp = 1.0 / static_cast<double>(cfg.temperature);
  const double max_scaled = static_cast<double>(order[0].first) * inv_temp;
  std::vector<double> probs(order.size());
  double sum_exp = 0.0;
  for (size_t i = 0; i < order.size(); ++i) {
    probs[i] = std::exp(static_cast<double>(order[i].first) * inv_temp - max_scaled);
    sum_exp += probs[i];
  }
  for (double& p : probs) p /= sum_exp;

  size_t nucleus_n = order.size();
  if (cfg.top_p > 0.0f && cfg.top_p < 1.0f) {
    double cum = 0.0;
    for (size_t i = 0; i < probs.size(); ++i) {
      cum += probs[i];
      if (cum >= static_cast<double>(cfg.top_p)) {
        nucleus_n = i + 1u;
        break;
      }
    }
  }
  double kept_sum = 0.0;
  for (size_t i = 0; i < nucleus_n; ++i) kept_sum += probs[i];
  result.reserve(nucleus_n);
  for (size_t i = 0; i < nucleus_n; ++i) {
    result.push_back(TokenProb{order[i].second, probs[i] / kept_sum});
  }
  return result;
}

inline int32_t sample_from_logits(
    const float* logits, uint32_t vocab_size, const SamplerConfig& cfg, std::mt19937_64& rng) {
  const auto dist = build_sampling_distribution(logits, vocab_size, cfg);
  if (dist.empty()) return -1;
  if (dist.size() == 1u) return dist[0].token_id;
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  const double u = uni(rng);
  double cum = 0.0;
  for (size_t i = 0; i + 1u < dist.size(); ++i) {
    cum += dist[i].prob;
    if (u < cum) return dist[i].token_id;
  }
  return dist.back().token_id;
}

inline void fake_quantize_a16_vec(const float* src, uint32_t len, float* dst) noexcept {
  float abs_max = 0.0f;
  for (uint32_t i = 0; i < len; ++i) abs_max = std::max(abs_max, std::fabs(src[i]));
  const float s_tok = std::max(abs_max / 32767.0f, 1e-12f);
  for (uint32_t i = 0; i < len; ++i) {
    dst[i] = std::max(-32767.0f, std::min(32767.0f, std::nearbyintf(src[i] / s_tok))) * s_tok;
  }
}

inline void fake_quantize_w8_matrix(const float* src, uint32_t rows, uint32_t cols, float* dst) {
  std::vector<int8_t> row_q(cols);
  for (uint32_t r = 0; r < rows; ++r) {
    float scale = 0.0f;
    quantize_row_sym_i8(src + static_cast<size_t>(r) * cols, cols, row_q.data(), &scale);
    float* dst_row = dst + static_cast<size_t>(r) * cols;
    for (uint32_t c = 0; c < cols; ++c) dst_row[c] = static_cast<float>(row_q[c]) * scale;
  }
}

inline void apply_rope_inplace(
    float* heads, uint32_t num_heads, uint32_t head_dim, uint32_t pos, float rope_theta) noexcept {
  const uint32_t half = head_dim / 2u;
  for (uint32_t j = 0; j < half; ++j) {
    const double angle = static_cast<double>(pos) *
                         std::pow(static_cast<double>(rope_theta), -2.0 * j / head_dim);
    const float c = static_cast<float>(std::cos(angle)), s = static_cast<float>(std::sin(angle));
    for (uint32_t h = 0; h < num_heads; ++h) {
      float* vec = heads + static_cast<size_t>(h) * head_dim;
      const float x0 = vec[j], x1 = vec[j + half];
      vec[j] = x0 * c - x1 * s;
      vec[j + half] = x1 * c + x0 * s;
    }
  }
}

struct BlockConfig {
  uint32_t hidden_size = 64;
  uint32_t n_heads = 4;
  uint32_t n_kv_heads = 2;
  uint32_t head_dim = 16;
  uint32_t intermediate_size = 128;
  uint32_t max_seq_len = 256;
  float rms_eps = 1e-6f;
  float rope_theta = 10000.0f;
  bool w8a16_quant = false;
};

struct SingleSeqKvCache {
  uint32_t max_seq_len = 0, n_kv_heads = 0, head_dim = 0, seq_len = 0;
  std::vector<float> k, v;

  void init(uint32_t max_len, uint32_t nkv, uint32_t dim) {
    max_seq_len = max_len;
    n_kv_heads = nkv;
    head_dim = dim;
    seq_len = 0;
    const size_t total = static_cast<size_t>(max_len) * nkv * dim;
    k.assign(total, 0.0f);
    v.assign(total, 0.0f);
  }
  void clear() noexcept { seq_len = 0; }
};

struct BlockWeights {
  std::vector<float> rms_attn, rms_mlp, w_q, w_k, w_v, w_o, w_gate, w_up, w_down;
};

class TransformerBlockRef {
 public:
  TransformerBlockRef(const BlockConfig& cfg, BlockWeights weights)
      : cfg_(cfg), weights_(std::move(weights)) {
    if (cfg_.hidden_size == 0 || cfg_.n_heads == 0 || cfg_.n_kv_heads == 0 || cfg_.head_dim == 0 ||
        (cfg_.head_dim % 2u) != 0u || (cfg_.n_heads % cfg_.n_kv_heads) != 0u ||
        cfg_.intermediate_size == 0 || cfg_.max_seq_len == 0) {
      throw std::invalid_argument("Invalid BlockConfig dimensions");
    }
    const size_t H = cfg_.hidden_size, QD = static_cast<size_t>(cfg_.n_heads) * cfg_.head_dim;
    const size_t KVD = static_cast<size_t>(cfg_.n_kv_heads) * cfg_.head_dim, I = cfg_.intermediate_size;
    if (weights_.rms_attn.size() != H || weights_.rms_mlp.size() != H ||
        weights_.w_q.size() != QD * H || weights_.w_k.size() != KVD * H ||
        weights_.w_v.size() != KVD * H || weights_.w_o.size() != H * QD ||
        weights_.w_gate.size() != I * H || weights_.w_up.size() != I * H ||
        weights_.w_down.size() != H * I) {
      throw std::invalid_argument("BlockWeights size mismatch against BlockConfig");
    }
    if (cfg_.w8a16_quant) {
      fake_quantize_w8_matrix(weights_.w_q.data(), static_cast<uint32_t>(QD), cfg_.hidden_size, weights_.w_q.data());
      fake_quantize_w8_matrix(weights_.w_k.data(), static_cast<uint32_t>(KVD), cfg_.hidden_size, weights_.w_k.data());
      fake_quantize_w8_matrix(weights_.w_v.data(), static_cast<uint32_t>(KVD), cfg_.hidden_size, weights_.w_v.data());
      fake_quantize_w8_matrix(weights_.w_o.data(), cfg_.hidden_size, static_cast<uint32_t>(QD), weights_.w_o.data());
      fake_quantize_w8_matrix(weights_.w_gate.data(), cfg_.intermediate_size, cfg_.hidden_size, weights_.w_gate.data());
      fake_quantize_w8_matrix(weights_.w_up.data(), cfg_.intermediate_size, cfg_.hidden_size, weights_.w_up.data());
      fake_quantize_w8_matrix(weights_.w_down.data(), cfg_.hidden_size, cfg_.intermediate_size, weights_.w_down.data());
    }
  }

  const BlockConfig& config() const noexcept { return cfg_; }

  void forward_step(const float* x_in, SingleSeqKvCache& cache, float* y_out) const {
    if (cache.seq_len >= cache.max_seq_len || cache.n_kv_heads != cfg_.n_kv_heads ||
        cache.head_dim != cfg_.head_dim) {
      throw std::runtime_error("SingleSeqKvCache capacity exceeded or uninitialized");
    }
    const uint32_t H = cfg_.hidden_size, NH = cfg_.n_heads, NKV = cfg_.n_kv_heads;
    const uint32_t D = cfg_.head_dim, QD = NH * D, KVD = NKV * D, I = cfg_.intermediate_size;
    const uint32_t pos = cache.seq_len;

    std::vector<float> xn(H), q(QD), k(KVD), v(KVD);
    rms_norm_f32(x_in, weights_.rms_attn.data(), H, cfg_.rms_eps, xn.data());
    linear_proj(xn.data(), weights_.w_q.data(), QD, H, q.data());
    linear_proj(xn.data(), weights_.w_k.data(), KVD, H, k.data());
    linear_proj(xn.data(), weights_.w_v.data(), KVD, H, v.data());

    apply_rope_inplace(q.data(), NH, D, pos, cfg_.rope_theta);
    apply_rope_inplace(k.data(), NKV, D, pos, cfg_.rope_theta);

    const size_t kv_off = static_cast<size_t>(pos) * KVD;
    std::memcpy(cache.k.data() + kv_off, k.data(), KVD * sizeof(float));
    std::memcpy(cache.v.data() + kv_off, v.data(), KVD * sizeof(float));
    cache.seq_len = pos + 1u;

    const uint32_t group_size = NH / NKV;
    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(D));
    std::vector<float> attn_concat(QD, 0.0f);
    std::vector<double> scores(cache.seq_len);

    for (uint32_t qh = 0; qh < NH; ++qh) {
      const uint32_t kvh = qh / group_size;
      const float* q_vec = q.data() + static_cast<size_t>(qh) * D;
      double max_score = -std::numeric_limits<double>::infinity();
      for (uint32_t t = 0; t < cache.seq_len; ++t) {
        const float* k_vec = cache.k.data() + (static_cast<size_t>(t) * NKV + kvh) * D;
        double dot = 0.0;
        for (uint32_t d = 0; d < D; ++d) dot += static_cast<double>(q_vec[d]) * k_vec[d];
        scores[t] = dot * static_cast<double>(inv_sqrt_d);
        if (scores[t] > max_score) max_score = scores[t];
      }
      double sum_exp = 0.0;
      for (uint32_t t = 0; t < cache.seq_len; ++t) {
        scores[t] = std::exp(scores[t] - max_score);
        sum_exp += scores[t];
      }
      float* out_vec = attn_concat.data() + static_cast<size_t>(qh) * D;
      for (uint32_t t = 0; t < cache.seq_len; ++t) {
        const float prob = static_cast<float>(scores[t] / sum_exp);
        const float* v_vec = cache.v.data() + (static_cast<size_t>(t) * NKV + kvh) * D;
        for (uint32_t d = 0; d < D; ++d) out_vec[d] += prob * v_vec[d];
      }
    }

    std::vector<float> attn_proj(H), h_mid(H), mlp_in(H), gate(I), up(I), fused(I), mlp_out(H);
    linear_proj(attn_concat.data(), weights_.w_o.data(), H, QD, attn_proj.data());
    for (uint32_t d = 0; d < H; ++d) h_mid[d] = x_in[d] + attn_proj[d];

    rms_norm_f32(h_mid.data(), weights_.rms_mlp.data(), H, cfg_.rms_eps, mlp_in.data());
    linear_proj(mlp_in.data(), weights_.w_gate.data(), I, H, gate.data());
    linear_proj(mlp_in.data(), weights_.w_up.data(), I, H, up.data());
    for (uint32_t i = 0; i < I; ++i) {
      fused[i] = (gate[i] / (1.0f + std::exp(-gate[i]))) * up[i];
    }
    linear_proj(fused.data(), weights_.w_down.data(), H, I, mlp_out.data());
    for (uint32_t d = 0; d < H; ++d) y_out[d] = h_mid[d] + mlp_out[d];
  }

 private:
  void linear_proj(
      const float* x, const float* w, uint32_t out_dim, uint32_t in_dim, float* y) const {
    std::vector<float> x_quant;
    const float* x_use = x;
    if (cfg_.w8a16_quant) {
      x_quant.resize(in_dim);
      fake_quantize_a16_vec(x, in_dim, x_quant.data());
      x_use = x_quant.data();
    }
    for (uint32_t o = 0; o < out_dim; ++o) {
      const float* w_row = w + static_cast<size_t>(o) * in_dim;
      double acc = 0.0;
      for (uint32_t i = 0; i < in_dim; ++i) acc += static_cast<double>(w_row[i]) * x_use[i];
      y[o] = static_cast<float>(acc);
    }
  }

  BlockConfig cfg_;
  BlockWeights weights_;
};

}  // namespace rt_ops
