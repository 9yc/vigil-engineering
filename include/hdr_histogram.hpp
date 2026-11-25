#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace hdr {

struct CdfBucket {
    double value_low = 0.0;
    double value_high = 0.0;
    uint64_t count = 0;
    uint64_t cumulative_count = 0;
    double cdf = 0.0;
};

struct RequestSpan {
    double arrival_s = 0.0;
    double duration_ms = 0.0;
};

struct WindowConcurrencyReport {
    size_t sample_count = 0;
    double span_s = 0.0;
    double mean_arrival_rps = 0.0;
    double mean_service_s = 0.0;
    double littles_law_concurrency = 0.0;
    double time_weighted_concurrency = 0.0;
    uint64_t peak_concurrency = 0;
    double window_s = 1.0;
    double peak_arrival_window_rps = 0.0;
    double peak_completion_window_rps = 0.0;
};

namespace detail {
inline int clz_u64(uint64_t x) noexcept {
    if (x == 0) return 64;
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_clzll(x);
#else
    int n = 0;
    for (int s = 32; s > 0; s >>= 1) {
        if ((x >> (64 - s)) == 0) { n += s; x <<= s; }
    }
    return n;
#endif
}
inline int floor_log2_u64(uint64_t x) noexcept { return 63 - clz_u64(x); }
}  // namespace detail

class HdrHistogram {
public:
    HdrHistogram(int64_t lowest_discernible_value,
                 int64_t highest_trackable_value,
                 int significant_figures,
                 bool clamp_overflow = true)
        : lowest_discernible_value_(lowest_discernible_value),
          highest_trackable_value_(highest_trackable_value),
          significant_figures_(significant_figures),
          clamp_overflow_(clamp_overflow) {
        if (lowest_discernible_value < 1 || significant_figures < 1 || significant_figures > 5 ||
            highest_trackable_value < 2 * lowest_discernible_value) {
            throw std::invalid_argument("Invalid HdrHistogram range or significant_figures");
        }
        int mag = static_cast<int>(std::ceil(std::log2(2.0 * std::pow(10.0, significant_figures))));
        sub_bucket_half_mag_ = std::max(0, mag - 1);
        unit_magnitude_ = detail::floor_log2_u64(static_cast<uint64_t>(lowest_discernible_value));
        if (sub_bucket_half_mag_ + unit_magnitude_ > 61) {
            throw std::invalid_argument("Precision + unit_magnitude overflows 64-bit integer");
        }
        sub_bucket_count_ = 1 << (sub_bucket_half_mag_ + 1);
        sub_bucket_half_count_ = sub_bucket_count_ / 2;
        sub_bucket_mask_ = (static_cast<uint64_t>(sub_bucket_count_) - 1ULL) << unit_magnitude_;

        int64_t smallest_untrackable = static_cast<int64_t>(sub_bucket_count_) << unit_magnitude_;
        int buckets = 1;
        while (smallest_untrackable <= highest_trackable_value_) {
            if (smallest_untrackable > (std::numeric_limits<int64_t>::max() >> 1)) { ++buckets; break; }
            smallest_untrackable <<= 1;
            ++buckets;
        }
        bucket_count_ = buckets;
        counts_.assign(static_cast<size_t>(bucket_count_ + 1) * static_cast<size_t>(sub_bucket_half_count_), 0ULL);
        reset();
    }

    void reset() noexcept {
        std::fill(counts_.begin(), counts_.end(), 0ULL);
        total_count_ = 0;
        overflow_count_ = 0;
        min_value_ = std::numeric_limits<int64_t>::max();
        max_value_ = 0;
    }

    bool record_value(int64_t value, uint64_t count = 1) {
        if (value < 0) throw std::invalid_argument("Negative value rejected");
        if (count == 0) return true;
        int64_t adj = value;
        if (value > highest_trackable_value_) {
            overflow_count_ += count;
            if (!clamp_overflow_) return false;
            adj = highest_trackable_value_;
        }
        size_t idx = std::min(counts_index_for(adj), counts_.size() - 1);
        counts_[idx] += count;
        total_count_ += count;
        min_value_ = std::min(min_value_, adj);
        max_value_ = std::max(max_value_, adj);
        return value <= highest_trackable_value_;
    }

    void merge(const HdrHistogram& other) {
        ensure_compatible(other);
        for (size_t i = 0; i < counts_.size(); ++i) counts_[i] += other.counts_[i];
        total_count_ += other.total_count_;
        overflow_count_ += other.overflow_count_;
        if (other.total_count_ > 0) {
            min_value_ = std::min(min_value_, other.min_value_);
            max_value_ = std::max(max_value_, other.max_value_);
        }
    }

    void subtract(const HdrHistogram& other) {
        ensure_compatible(other);
        if (other.overflow_count_ > overflow_count_) throw std::underflow_error("Overflow count underflow");
        for (size_t i = 0; i < counts_.size(); ++i) {
            if (other.counts_[i] > counts_[i]) throw std::underflow_error("Bucket count underflow");
        }
        for (size_t i = 0; i < counts_.size(); ++i) counts_[i] -= other.counts_[i];
        total_count_ -= other.total_count_;
        overflow_count_ -= other.overflow_count_;
        recompute_extrema();
    }

    uint64_t total_count() const noexcept { return total_count_; }
    uint64_t overflow_count() const noexcept { return overflow_count_; }
    int significant_figures() const noexcept { return significant_figures_; }
    int64_t lowest_discernible_value() const noexcept { return lowest_discernible_value_; }
    int64_t highest_trackable_value() const noexcept { return highest_trackable_value_; }
    int64_t min_value() const noexcept { return total_count_ == 0 ? 0 : min_value_; }
    int64_t max_value() const noexcept { return total_count_ == 0 ? 0 : max_value_; }

    double value_at_percentile(double pct) const {
        if (!std::isfinite(pct) || pct < 0.0 || pct > 100.0) {
            throw std::invalid_argument("percentile must be in [0.0, 100.0]");
        }
        if (total_count_ == 0) return 0.0;
        if (pct == 0.0) return static_cast<double>(min_value_);
        double rank = (pct / 100.0) * static_cast<double>(total_count_);
        uint64_t target = std::clamp<uint64_t>(
            static_cast<uint64_t>(std::ceil(std::max(0.0, rank - 1e-9))), 1ULL, total_count_);
        uint64_t running = 0;
        for (size_t i = 0; i < counts_.size(); ++i) {
            uint64_t c = counts_[i];
            if (c == 0) continue;
            if (running + c >= target) {
                int64_t low = value_from_index(i), high = highest_equivalent_value(low);
                double pos = std::clamp((rank - static_cast<double>(running) - 0.5) / static_cast<double>(c), 0.0, 1.0);
                double est = static_cast<double>(low) + static_cast<double>(high - low) * pos;
                return std::clamp(est, static_cast<double>(min_value_), static_cast<double>(max_value_));
            }
            running += c;
        }
        return static_cast<double>(max_value_);
    }

    double mean() const noexcept {
        if (total_count_ == 0) return 0.0;
        long double sum = 0.0L;
        for (size_t i = 0; i < counts_.size(); ++i) {
            if (counts_[i] != 0) {
                sum += static_cast<long double>(counts_[i]) *
                       static_cast<long double>(median_equivalent_value(value_from_index(i)));
            }
        }
        return static_cast<double>(sum / static_cast<long double>(total_count_));
    }

    double stddev() const noexcept {
        if (total_count_ == 0) return 0.0;
        long double m = static_cast<long double>(mean()), acc = 0.0L;
        for (size_t i = 0; i < counts_.size(); ++i) {
            if (counts_[i] != 0) {
                long double d = static_cast<long double>(median_equivalent_value(value_from_index(i))) - m;
                acc += static_cast<long double>(counts_[i]) * d * d;
            }
        }
        return static_cast<double>(std::sqrt(acc / static_cast<long double>(total_count_)));
    }

    std::vector<CdfBucket> export_cdf(double unit_divisor = 1.0, size_t max_buckets = 0) const {
        if (!std::isfinite(unit_divisor) || unit_divisor <= 0.0) {
            throw std::invalid_argument("unit_divisor must be finite and > 0");
        }
        std::vector<CdfBucket> raw;
        if (total_count_ == 0) return raw;
        uint64_t cum = 0;
        for (size_t i = 0; i < counts_.size(); ++i) {
            if (counts_[i] == 0) continue;
            cum += counts_[i];
            int64_t low = value_from_index(i), high = highest_equivalent_value(low);
            raw.push_back({static_cast<double>(low) / unit_divisor, static_cast<double>(high) / unit_divisor,
                           counts_[i], cum, static_cast<double>(cum) / static_cast<double>(total_count_)});
        }
        if (max_buckets == 0 || raw.size() <= max_buckets) return raw;
        std::vector<CdfBucket> out;
        out.reserve(max_buckets);
        for (size_t g = 0; g < max_buckets; ++g) {
            size_t b = (g * raw.size()) / max_buckets, e = ((g + 1) * raw.size()) / max_buckets;
            if (b >= e) continue;
            uint64_t c = 0;
            for (size_t k = b; k < e; ++k) c += raw[k].count;
            out.push_back({raw[b].value_low, raw[e - 1].value_high, c, raw[e - 1].cumulative_count, raw[e - 1].cdf});
        }
        return out;
    }

    int64_t lowest_equivalent_value(int64_t v) const noexcept {
        int b = get_bucket_index(v);
        return value_from_location(b, get_sub_bucket_index(v, b));
    }

    int64_t highest_equivalent_value(int64_t v) const noexcept {
        int b = get_bucket_index(v), s = get_sub_bucket_index(v, b);
        return lowest_equivalent_value(v) + (1LL << (unit_magnitude_ + ((s >= sub_bucket_count_) ? b + 1 : b))) - 1;
    }

    double median_equivalent_value(int64_t v) const noexcept {
        return 0.5 * (static_cast<double>(lowest_equivalent_value(v)) + static_cast<double>(highest_equivalent_value(v)));
    }

private:
    int get_bucket_index(int64_t v) const noexcept {
        return 64 - unit_magnitude_ - (sub_bucket_half_mag_ + 1) -
               detail::clz_u64(static_cast<uint64_t>(v) | sub_bucket_mask_);
    }
    int get_sub_bucket_index(int64_t v, int b) const noexcept {
        return static_cast<int>(static_cast<uint64_t>(v) >> (b + unit_magnitude_));
    }
    size_t counts_index_for(int64_t v) const noexcept {
        int b = get_bucket_index(v), s = get_sub_bucket_index(v, b);
        return static_cast<size_t>(((b + 1) << sub_bucket_half_mag_) + s - sub_bucket_half_count_);
    }
    int64_t value_from_location(int b, int s) const noexcept {
        return static_cast<int64_t>(static_cast<uint64_t>(s) << (b + unit_magnitude_));
    }
    int64_t value_from_index(size_t index) const noexcept {
        int b = static_cast<int>(index >> sub_bucket_half_mag_) - 1;
        int s = static_cast<int>(index & static_cast<size_t>(sub_bucket_half_count_ - 1));
        if (b < 0) b = 0; else s += sub_bucket_half_count_;
        return value_from_location(b, s);
    }
    void ensure_compatible(const HdrHistogram& o) const {
        if (lowest_discernible_value_ != o.lowest_discernible_value_ ||
            highest_trackable_value_ != o.highest_trackable_value_ ||
            significant_figures_ != o.significant_figures_) {
            throw std::invalid_argument("Incompatible HdrHistogram configuration");
        }
    }
    void recompute_extrema() noexcept {
        if (total_count_ == 0) {
            min_value_ = std::numeric_limits<int64_t>::max();
            max_value_ = 0;
            return;
        }
        for (size_t i = 0; i < counts_.size(); ++i) {
            if (counts_[i] != 0) { min_value_ = value_from_index(i); break; }
        }
        for (size_t i = counts_.size(); i > 0; --i) {
            if (counts_[i - 1] != 0) { max_value_ = highest_equivalent_value(value_from_index(i - 1)); break; }
        }
    }

    int64_t lowest_discernible_value_;
    int64_t highest_trackable_value_;
    int significant_figures_;
    bool clamp_overflow_;
    int unit_magnitude_ = 0, sub_bucket_half_mag_ = 0, sub_bucket_count_ = 0, sub_bucket_half_count_ = 0, bucket_count_ = 0;
    uint64_t sub_bucket_mask_ = 0, total_count_ = 0, overflow_count_ = 0;
    std::vector<uint64_t> counts_;
    int64_t min_value_ = std::numeric_limits<int64_t>::max(), max_value_ = 0;
};

class DoubleHdrHistogram {
public:
    DoubleHdrHistogram(double max_trackable_units = 3600000.0,
                       int significant_figures = 3,
                       double ticks_per_unit = 1000.0,
                       bool clamp_overflow = true)
        : ticks_per_unit_(ticks_per_unit),
          hist_(1, std::max<int64_t>(2, static_cast<int64_t>(std::ceil(max_trackable_units * ticks_per_unit))),
                significant_figures, clamp_overflow) {
        if (!std::isfinite(max_trackable_units) || max_trackable_units <= 0.0 ||
            !std::isfinite(ticks_per_unit) || ticks_per_unit <= 0.0) {
            throw std::invalid_argument("DoubleHdrHistogram scale bounds must be finite and > 0");
        }
    }

    bool record(double value, uint64_t count = 1) {
        if (!std::isfinite(value) || value < 0.0) throw std::invalid_argument("Value must be finite and >= 0");
        double scaled = value * ticks_per_unit_;
        int64_t lim = std::numeric_limits<int64_t>::max() / 2;
        int64_t ticks = scaled >= static_cast<double>(lim) ? lim : static_cast<int64_t>(std::llround(scaled));
        uint64_t prev_tot = hist_.total_count();
        bool ok = hist_.record_value(ticks, count);
        if (hist_.total_count() > prev_tot) {
            double v = ok ? value : max_trackable();
            raw_min_ = std::min(raw_min_, v);
            raw_max_ = std::max(raw_max_, v);
            long double lv = static_cast<long double>(v);
            exact_sum_ += lv * static_cast<long double>(count);
            exact_sum_sq_ += lv * lv * static_cast<long double>(count);
        }
        return ok;
    }

    void merge(const DoubleHdrHistogram& o) {
        hist_.merge(o.hist_);
        if (o.total_count() > 0) {
            raw_min_ = std::min(raw_min_, o.raw_min_);
            raw_max_ = std::max(raw_max_, o.raw_max_);
        }
        exact_moments_ = exact_moments_ && o.exact_moments_;
        exact_sum_ += o.exact_sum_;
        exact_sum_sq_ += o.exact_sum_sq_;
    }

    void subtract(const DoubleHdrHistogram& o) {
        hist_.subtract(o.hist_);
        raw_min_ = static_cast<double>(hist_.min_value()) / ticks_per_unit_;
        raw_max_ = static_cast<double>(hist_.max_value()) / ticks_per_unit_;
        exact_moments_ = false;
    }

    uint64_t total_count() const noexcept { return hist_.total_count(); }
    uint64_t overflow_count() const noexcept { return hist_.overflow_count(); }
    double max_trackable() const noexcept { return static_cast<double>(hist_.highest_trackable_value()) / ticks_per_unit_; }
    double min() const noexcept { return total_count() == 0 ? 0.0 : raw_min_; }
    double max() const noexcept { return total_count() == 0 ? 0.0 : raw_max_; }
    double mean() const noexcept {
        if (total_count() == 0) return 0.0;
        return exact_moments_ ? static_cast<double>(exact_sum_ / static_cast<long double>(total_count()))
                              : hist_.mean() / ticks_per_unit_;
    }
    double stddev() const noexcept {
        if (total_count() == 0) return 0.0;
        if (!exact_moments_) return hist_.stddev() / ticks_per_unit_;
        long double n = static_cast<long double>(total_count()), m = exact_sum_ / n;
        return static_cast<double>(std::sqrt(std::max(0.0L, (exact_sum_sq_ / n) - m * m)));
    }
    double percentile(double pct) const { return hist_.value_at_percentile(pct) / ticks_per_unit_; }
    std::vector<CdfBucket> export_cdf(size_t max_buckets = 0) const { return hist_.export_cdf(ticks_per_unit_, max_buckets); }

private:
    double ticks_per_unit_;
    HdrHistogram hist_;
    double raw_min_ = std::numeric_limits<double>::infinity(), raw_max_ = 0.0;
    bool exact_moments_ = true;
    long double exact_sum_ = 0.0L, exact_sum_sq_ = 0.0L;
};

inline WindowConcurrencyReport analyze_sliding_window_and_littles_law(
    const std::vector<RequestSpan>& spans, double window_s) {
    if (!std::isfinite(window_s) || window_s <= 0.0) throw std::invalid_argument("window_s must be > 0");
    WindowConcurrencyReport out;
    out.window_s = window_s;
    if (spans.empty()) return out;

    std::vector<double> arrivals, completions;
    std::vector<std::pair<double, int>> events;
    arrivals.reserve(spans.size());
    completions.reserve(spans.size());
    events.reserve(spans.size() * 2);

    double min_arr = std::numeric_limits<double>::infinity(), max_comp = -std::numeric_limits<double>::infinity();
    long double sum_s = 0.0L;
    for (const auto& r : spans) {
        if (!std::isfinite(r.arrival_s) || !std::isfinite(r.duration_ms) || r.duration_ms < 0.0) {
            throw std::invalid_argument("Invalid span timestamp or negative duration");
        }
        double dur_s = r.duration_ms * 1e-3, comp_s = r.arrival_s + dur_s;
        arrivals.push_back(r.arrival_s);
        completions.push_back(comp_s);
        events.emplace_back(r.arrival_s, +1);
        events.emplace_back(comp_s, -1);
        min_arr = std::min(min_arr, r.arrival_s);
        max_comp = std::max(max_comp, comp_s);
        sum_s += static_cast<long double>(dur_s);
    }

    std::sort(arrivals.begin(), arrivals.end());
    std::sort(completions.begin(), completions.end());
    std::sort(events.begin(), events.end(), [](const auto& a, const auto& b) {
        return a.first != b.first ? (a.first < b.first) : (a.second > b.second);
    });

    out.sample_count = spans.size();
    out.mean_service_s = static_cast<double>(sum_s / static_cast<long double>(spans.size()));
    out.span_s = std::max(0.0, max_comp - min_arr);

    int64_t active = 0, peak = 0;
    long double area = 0.0L;
    double prev_t = events.front().first;
    for (const auto& ev : events) {
        if (ev.first > prev_t && active > 0) area += static_cast<long double>(active) * static_cast<long double>(ev.first - prev_t);
        active += ev.second;
        peak = std::max(peak, active);
        prev_t = ev.first;
    }
    out.peak_concurrency = static_cast<uint64_t>(std::max<int64_t>(0, peak));
    if (out.span_s > 0.0) {
        out.mean_arrival_rps = static_cast<double>(spans.size()) / out.span_s;
        out.littles_law_concurrency = out.mean_arrival_rps * out.mean_service_s;
        out.time_weighted_concurrency = static_cast<double>(area / static_cast<long double>(out.span_s));
    }

    auto peak_rate = [window_s](const std::vector<double>& ts) {
        size_t best = 0, r = 0;
        for (size_t l = 0; l < ts.size(); ++l) {
            while (r < ts.size() && ts[r] <= ts[l] + window_s + 1e-12) ++r;
            best = std::max(best, r - l);
        }
        return static_cast<double>(best) / window_s;
    };
    out.peak_arrival_window_rps = peak_rate(arrivals);
    out.peak_completion_window_rps = peak_rate(completions);
    return out;
}

}  // namespace hdr
