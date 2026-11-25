#ifndef THERMAL_COOLDOWN_HPP_
#define THERMAL_COOLDOWN_HPP_

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace thermal_cooldown {

struct ThermalSample { double time_s = 0.0; double temp_c = 0.0; };

struct BatchThermalRecord {
    std::string batch_id;
    double start_time_s = 0.0, end_time_s = 0.0, start_temp_c = 0.0, end_temp_c = 0.0;
};

struct CoolingFitResult {
    bool converged = false;
    double ref_time_s = 0.0, t_amb_c = 0.0, t_initial_c = 0.0, tau_s = 0.0;
    double rmse_c = 0.0, max_abs_error_c = 0.0, r_squared = 0.0;
    int iterations = 0;
};

struct TargetCooldownEstimate {
    bool requested = false, already_at_or_below_target = false, reachable = false, empirical_crossed = false;
    double target_temp_c = 0.0, predicted_elapsed_from_start_s = 0.0;
    double predicted_target_time_s = 0.0, predicted_remaining_from_end_s = 0.0, empirical_crossing_time_s = 0.0;
};

struct PlateauConfig { double hold_s = 30.0, max_slope_c_per_s = 0.02, max_span_c = 0.5; };

struct PlateauReport {
    bool detected = false, settled_at_end = false;
    double onset_time_s = 0.0, window_end_time_s = 0.0, mean_temp_c = 0.0, slope_c_per_s = 0.0, span_c = 0.0;
    double trailing_mean_temp_c = 0.0, trailing_slope_c_per_s = 0.0, trailing_span_c = 0.0;
};

struct BatchDriftThresholds {
    double max_total_start_drift_c = 1.5, max_slope_c_per_batch = 0.35, max_step_start_rise_c = 1.0;
};

struct BatchDriftReport {
    size_t batch_count = 0;
    double initial_start_temp_c = 0.0, final_start_temp_c = 0.0, total_start_drift_c = 0.0, total_end_drift_c = 0.0;
    double start_slope_c_per_batch = 0.0, end_slope_c_per_batch = 0.0;
    double max_step_start_rise_c = 0.0, mean_intra_batch_rise_c = 0.0;
    bool thermal_creep_detected = false;
    std::vector<std::string> creep_reasons;
};

inline void write_exclusive_text(const std::string& path, const std::string& content) {
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (fd < 0) {
        throw std::runtime_error("exclusive open rejected for '" + path + "': " + std::string(std::strerror(errno)));
    }
    for (size_t off = 0; off < content.size();) {
        const ssize_t n = ::write(fd, content.data() + off, content.size() - off);
        if (n < 0) {
            if (errno == EINTR) continue;
            const int err = errno;
            (void)::close(fd);
            throw std::runtime_error("write failed for '" + path + "': " + std::string(std::strerror(err)));
        }
        off += static_cast<size_t>(n);
    }
    (void)::fsync(fd);
    if (::close(fd) != 0) throw std::runtime_error("close failed for '" + path + "'");
}

inline void validate_thermal_series(const std::vector<ThermalSample>& trace) {
    if (trace.size() < 3U) throw std::invalid_argument("Thermal trajectory requires at least 3 samples");
    for (size_t i = 0; i < trace.size(); ++i) {
        if (!std::isfinite(trace[i].time_s) || !std::isfinite(trace[i].temp_c)) {
            throw std::invalid_argument("Non-finite sample in thermal trajectory");
        }
        if (i > 0 && trace[i].time_s <= trace[i - 1U].time_s) {
            throw std::invalid_argument("Thermal trajectory timestamps must strictly increase");
        }
    }
}

inline bool solve_linear_3x3(std::array<std::array<double, 3>, 3> a,
                             std::array<double, 3> b,
                             std::array<double, 3>& x) noexcept {
    for (size_t col = 0; col < 3U; ++col) {
        size_t pivot = col;
        for (size_t row = col + 1U; row < 3U; ++row) {
            if (std::abs(a[row][col]) > std::abs(a[pivot][col])) pivot = row;
        }
        if (std::abs(a[pivot][col]) < 1e-15 || !std::isfinite(a[pivot][col])) return false;
        if (pivot != col) { std::swap(a[col], a[pivot]); std::swap(b[col], b[pivot]); }
        for (size_t row = col + 1U; row < 3U; ++row) {
            const double f = a[row][col] / a[col][col];
            for (size_t k = col; k < 3U; ++k) a[row][k] -= f * a[col][k];
            b[row] -= f * b[col];
        }
    }
    for (int r = 2; r >= 0; --r) {
        const size_t ur = static_cast<size_t>(r);
        double sum = b[ur];
        for (size_t k = ur + 1U; k < 3U; ++k) sum -= a[ur][k] * x[k];
        x[ur] = sum / a[ur][ur];
        if (!std::isfinite(x[ur])) return false;
    }
    return true;
}

inline CoolingFitResult fit_newton_cooling(const std::vector<ThermalSample>& trace,
                                           bool fix_ambient = false,
                                           double fixed_ambient_c = 0.0) {
    validate_thermal_series(trace);
    const size_t n = trace.size();
    const double t_ref = trace.front().time_s;
    const double span_s = trace.back().time_s - t_ref;
    if (span_s <= 0.0) throw std::invalid_argument("Zero duration thermal trajectory");

    std::vector<double> dt(n);
    double y_mean = 0.0;
    for (size_t i = 0; i < n; ++i) {
        dt[i] = trace[i].time_s - t_ref;
        y_mean += trace[i].temp_c;
    }
    y_mean /= static_cast<double>(n);

    auto eval_sse = [&](double t_amb, double t0, double tau, double* max_err_out = nullptr) -> double {
        const double amp = t0 - t_amb;
        double sse = 0.0;
        double max_e = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double pred = t_amb + amp * std::exp(-dt[i] / tau);
            const double err = pred - trace[i].temp_c;
            sse += err * err;
            max_e = std::max(max_e, std::abs(err));
        }
        if (max_err_out != nullptr) *max_err_out = max_e;
        return sse;
    };

    // Stage 1: Exact conditional linear least-squares over a log-spaced tau grid.
    const double tau_min = std::max(1e-3, (trace[1].time_s - trace[0].time_s) * 0.25);
    const double tau_max = std::max(tau_min * 10.0, span_s * 20.0);
    constexpr int kGridSteps = 160;

    double best_sse = std::numeric_limits<double>::infinity();
    double best_amb = fix_ambient ? fixed_ambient_c : trace.back().temp_c;
    double best_t0 = trace.front().temp_c;
    double best_tau = span_s * 0.5;

    for (int g = 0; g < kGridSteps; ++g) {
        const double frac = static_cast<double>(g) / static_cast<double>(kGridSteps - 1);
        const double tau = tau_min * std::pow(tau_max / tau_min, frac);
        if (fix_ambient) {
            double s_xx = 0.0, s_xy = 0.0;
            for (size_t i = 0; i < n; ++i) {
                const double x = std::exp(-dt[i] / tau);
                s_xx += x * x;
                s_xy += x * (trace[i].temp_c - fixed_ambient_c);
            }
            if (s_xx <= 1e-18) continue;
            const double t0 = fixed_ambient_c + (s_xy / s_xx);
            const double sse = eval_sse(fixed_ambient_c, t0, tau);
            if (sse < best_sse) { best_sse = sse; best_amb = fixed_ambient_c; best_t0 = t0; best_tau = tau; }
        } else {
            double sx = 0.0, sxx = 0.0, sy = 0.0, sxy = 0.0;
            for (size_t i = 0; i < n; ++i) {
                const double x = std::exp(-dt[i] / tau), y = trace[i].temp_c;
                sx += x; sxx += x * x; sy += y; sxy += x * y;
            }
            const double dn = static_cast<double>(n), det = dn * sxx - sx * sx;
            if (std::abs(det) <= 1e-16) continue;
            const double t_amb = (sxx * sy - sx * sxy) / det;
            const double t0 = t_amb + (dn * sxy - sx * sy) / det;
            const double sse = eval_sse(t_amb, t0, tau);
            if (sse < best_sse) { best_sse = sse; best_amb = t_amb; best_t0 = t0; best_tau = tau; }
        }
    }

    // Stage 2: Damped Gauss-Newton (Levenberg-Marquardt) refinement.
    double lambda = 1e-4;
    bool converged = false;
    int iter_used = 0;

    for (int iter = 0; iter < 80; ++iter) {
        iter_used = iter + 1;
        std::array<std::array<double, 3>, 3> jtj{{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}};
        std::array<double, 3> neg_jtr{{0.0, 0.0, 0.0}};
        const double amp = best_t0 - best_amb;

        for (size_t i = 0; i < n; ++i) {
            const double e = std::exp(-dt[i] / best_tau);
            const double r = (best_amb + amp * e) - trace[i].temp_c;
            const std::array<double, 3> j{{fix_ambient ? 0.0 : (1.0 - e), e, amp * (dt[i] / (best_tau * best_tau)) * e}};
            for (size_t r_idx = 0; r_idx < 3U; ++r_idx) {
                neg_jtr[r_idx] -= j[r_idx] * r;
                for (size_t c_idx = 0; c_idx < 3U; ++c_idx) jtj[r_idx][c_idx] += j[r_idx] * j[c_idx];
            }
        }
        if (fix_ambient) { jtj[0][0] = 1.0; neg_jtr[0] = 0.0; }

        bool step_accepted = false;
        for (int trial = 0; trial < 12; ++trial) {
            auto damped = jtj;
            for (size_t d = 0; d < 3U; ++d) damped[d][d] += lambda * (jtj[d][d] + 1.0);
            if (fix_ambient) damped[0][0] = 1.0;
            std::array<double, 3> step{{0.0, 0.0, 0.0}};
            if (!solve_linear_3x3(damped, neg_jtr, step)) { lambda *= 10.0; continue; }
            const double cand_amb = fix_ambient ? fixed_ambient_c : (best_amb + step[0]);
            const double cand_t0 = best_t0 + step[1], cand_tau = best_tau + step[2];
            if (cand_tau <= tau_min * 0.1 || cand_tau > tau_max * 10.0) { lambda *= 10.0; continue; }
            const double cand_sse = eval_sse(cand_amb, cand_t0, cand_tau);
            if (cand_sse <= best_sse) {
                const double rel_step = (std::abs(step[0]) + std::abs(step[1]) + std::abs(step[2])) /
                                        (1.0 + std::abs(best_amb) + std::abs(best_t0) + std::abs(best_tau));
                best_amb = cand_amb; best_t0 = cand_t0; best_tau = cand_tau; best_sse = cand_sse;
                lambda = std::max(1e-12, lambda * 0.3);
                step_accepted = true;
                if (rel_step < 1e-10) converged = true;
                break;
            }
            lambda *= 8.0;
        }
        if (converged || !step_accepted) { converged = true; break; }
    }

    double max_abs_err = 0.0, sst = 0.0;
    const double final_sse = eval_sse(best_amb, best_t0, best_tau, &max_abs_err);
    for (const auto& s : trace) sst += (s.temp_c - y_mean) * (s.temp_c - y_mean);
    return CoolingFitResult{converged, t_ref, best_amb, best_t0, best_tau,
                            std::sqrt(final_sse / static_cast<double>(n)), max_abs_err,
                            sst > 1e-15 ? std::max(0.0, 1.0 - final_sse / sst) : 1.0, iter_used};
}

inline TargetCooldownEstimate estimate_cooldown_to_target(const std::vector<ThermalSample>& trace,
                                                          const CoolingFitResult& fit,
                                                          double target_temp_c) {
    TargetCooldownEstimate est{};
    est.requested = true;
    est.target_temp_c = target_temp_c;
    if (trace.front().temp_c <= target_temp_c) {
        est.empirical_crossed = true;
        est.empirical_crossing_time_s = trace.front().time_s;
    } else {
        for (size_t i = 1U; i < trace.size(); ++i) {
            if (trace[i].temp_c <= target_temp_c) {
                const double t0 = trace[i - 1U].time_s, t1 = trace[i].time_s;
                const double y0 = trace[i - 1U].temp_c, y1 = trace[i].temp_c;
                est.empirical_crossed = true;
                est.empirical_crossing_time_s = t0 + std::clamp((y0 - target_temp_c) / (y0 - y1), 0.0, 1.0) * (t1 - t0);
                break;
            }
        }
    }
    if (fit.t_initial_c <= target_temp_c) {
        est.already_at_or_below_target = true;
        est.reachable = true;
        est.predicted_target_time_s = fit.ref_time_s;
        return est;
    }
    if (target_temp_c <= fit.t_amb_c || fit.tau_s <= 0.0) return est;
    const double ratio = (target_temp_c - fit.t_amb_c) / (fit.t_initial_c - fit.t_amb_c);
    if (ratio <= 0.0 || ratio >= 1.0) return est;
    est.reachable = true;
    est.predicted_elapsed_from_start_s = -fit.tau_s * std::log(ratio);
    est.predicted_target_time_s = fit.ref_time_s + est.predicted_elapsed_from_start_s;
    est.predicted_remaining_from_end_s = std::max(0.0, est.predicted_target_time_s - trace.back().time_s);
    return est;
}

inline void window_ols_stats(const std::vector<ThermalSample>& trace,
                             size_t lo,
                             size_t hi,
                             double& mean_c,
                             double& slope_c_per_s,
                             double& span_c) noexcept {
    const size_t cnt = hi - lo + 1U;
    double t_bar = 0.0;
    double y_bar = 0.0;
    double y_min = trace[lo].temp_c;
    double y_max = trace[lo].temp_c;
    for (size_t k = lo; k <= hi; ++k) {
        t_bar += trace[k].time_s;
        y_bar += trace[k].temp_c;
        y_min = std::min(y_min, trace[k].temp_c);
        y_max = std::max(y_max, trace[k].temp_c);
    }
    t_bar /= static_cast<double>(cnt);
    y_bar /= static_cast<double>(cnt);

    double num = 0.0;
    double den = 0.0;
    for (size_t k = lo; k <= hi; ++k) {
        const double dt = trace[k].time_s - t_bar;
        num += dt * (trace[k].temp_c - y_bar);
        den += dt * dt;
    }
    mean_c = y_bar;
    slope_c_per_s = den > 1e-18 ? (num / den) : 0.0;
    span_c = y_max - y_min;
}

inline PlateauReport detect_steady_state_plateau(const std::vector<ThermalSample>& trace,
                                                 const PlateauConfig& cfg) {
    validate_thermal_series(trace);
    if (cfg.hold_s <= 0.0 || cfg.max_slope_c_per_s < 0.0 || cfg.max_span_c < 0.0) {
        throw std::invalid_argument("Invalid PlateauConfig thresholds");
    }

    PlateauReport rep{};
    const size_t n = trace.size();
    for (size_t i = 0; i < n; ++i) {
        size_t j = i + 1U;
        while (j < n && (trace[j].time_s - trace[i].time_s) < cfg.hold_s) ++j;
        if (j >= n) break;
        double mean_c = 0.0, slope = 0.0, span = 0.0;
        window_ols_stats(trace, i, j, mean_c, slope, span);
        if (std::abs(slope) <= cfg.max_slope_c_per_s && span <= cfg.max_span_c) {
            rep.detected = true;
            rep.onset_time_s = trace[i].time_s;
            rep.window_end_time_s = trace[j].time_s;
            rep.mean_temp_c = mean_c;
            rep.slope_c_per_s = slope;
            rep.span_c = span;
            break;
        }
    }

    // Evaluate trailing window anchored at trace end.
    const double end_t = trace.back().time_s;
    if ((end_t - trace.front().time_s) >= cfg.hold_s) {
        size_t tail_lo = n - 1U;
        while (tail_lo > 0 && (end_t - trace[tail_lo].time_s) < cfg.hold_s) --tail_lo;
        window_ols_stats(trace, tail_lo, n - 1U,
                         rep.trailing_mean_temp_c, rep.trailing_slope_c_per_s, rep.trailing_span_c);
        rep.settled_at_end = (std::abs(rep.trailing_slope_c_per_s) <= cfg.max_slope_c_per_s) &&
                             (rep.trailing_span_c <= cfg.max_span_c);
    }
    return rep;
}

inline BatchDriftReport audit_batch_thermal_drift(const std::vector<BatchThermalRecord>& batches,
                                                  const BatchDriftThresholds& limits) {
    if (batches.size() < 2U) {
        throw std::invalid_argument("Batch drift audit requires at least 2 consecutive batches");
    }
    BatchDriftReport rep{};
    rep.batch_count = batches.size();
    rep.initial_start_temp_c = batches.front().start_temp_c;
    rep.final_start_temp_c = batches.back().start_temp_c;
    rep.total_start_drift_c = batches.back().start_temp_c - batches.front().start_temp_c;
    rep.total_end_drift_c = batches.back().end_temp_c - batches.front().end_temp_c;

    const double n = static_cast<double>(batches.size());
    const double x_bar = 0.5 * (n - 1.0);
    double s_bar = 0.0, e_bar = 0.0, intra_sum = 0.0;
    double max_step = -std::numeric_limits<double>::infinity();

    for (size_t i = 0; i < batches.size(); ++i) {
        if (!std::isfinite(batches[i].start_temp_c) || !std::isfinite(batches[i].end_temp_c)) {
            throw std::invalid_argument("Non-finite temperature in batch record");
        }
        s_bar += batches[i].start_temp_c;
        e_bar += batches[i].end_temp_c;
        intra_sum += (batches[i].end_temp_c - batches[i].start_temp_c);
        if (i > 0) {
            max_step = std::max(max_step, batches[i].start_temp_c - batches[i - 1U].start_temp_c);
        }
    }
    s_bar /= n;
    e_bar /= n;
    rep.max_step_start_rise_c = max_step;
    rep.mean_intra_batch_rise_c = intra_sum / n;

    double num_s = 0.0, num_e = 0.0, den = 0.0;
    for (size_t i = 0; i < batches.size(); ++i) {
        const double dx = static_cast<double>(i) - x_bar;
        num_s += dx * (batches[i].start_temp_c - s_bar);
        num_e += dx * (batches[i].end_temp_c - e_bar);
        den += dx * dx;
    }
    rep.start_slope_c_per_batch = den > 0.0 ? (num_s / den) : 0.0;
    rep.end_slope_c_per_batch = den > 0.0 ? (num_e / den) : 0.0;

    if (rep.total_start_drift_c > limits.max_total_start_drift_c) {
        rep.creep_reasons.push_back("total_start_drift_exceeded");
    }
    if (rep.start_slope_c_per_batch > limits.max_slope_c_per_batch) {
        rep.creep_reasons.push_back("per_batch_start_slope_exceeded");
    }
    if (rep.max_step_start_rise_c > limits.max_step_start_rise_c) {
        rep.creep_reasons.push_back("step_start_rise_exceeded");
    }
    rep.thermal_creep_detected = !rep.creep_reasons.empty();
    return rep;
}

}  // namespace thermal_cooldown

#endif  // THERMAL_COOLDOWN_HPP_
