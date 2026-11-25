#ifndef CADENCE_CONTROLLER_HPP
#define CADENCE_CONTROLLER_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace cadence {

enum class ExecutionMode { Burst, FixedSleep, TokenBucket, ThermalFeedback };

inline std::string mode_to_string(ExecutionMode mode) {
    switch (mode) {
        case ExecutionMode::Burst: return "burst";
        case ExecutionMode::FixedSleep: return "fixed_sleep";
        case ExecutionMode::TokenBucket: return "token_bucket";
        case ExecutionMode::ThermalFeedback: return "thermal_feedback";
    }
    throw std::invalid_argument("unrecognized execution mode");
}

inline ExecutionMode parse_execution_mode(const std::string& raw) {
    if (raw == "burst") return ExecutionMode::Burst;
    if (raw == "fixed_sleep") return ExecutionMode::FixedSleep;
    if (raw == "token_bucket") return ExecutionMode::TokenBucket;
    if (raw == "thermal_feedback") return ExecutionMode::ThermalFeedback;
    throw std::invalid_argument("unknown execution mode: " + raw);
}

struct RequestSpec {
    std::string request_id;
    double arrival_s = 0.0;
    double compute_units = 0.0;
    double deadline_s = 0.0;
};

struct ThermalModelConfig {
    double ambient_temp_c = 25.0;
    double initial_temp_c = 25.0;
    double c_th_j_per_c = 12.0;
    double r_th_c_per_w = 4.5;
    double idle_power_w = 0.40;
    double nominal_power_w = 9.00;
    double nominal_rate_cu_per_s = 100.0;
    double throttle_temp_c = 55.0;
    double severe_throttle_temp_c = 65.0;
    double throttle_rate_scale = 0.45;
    double severe_rate_scale = 0.20;
    double power_exponent = 1.80;
    double throttle_leakage_w = 0.60;
    double integration_dt_s = 0.005;

    void validate() const {
        if (!std::isfinite(ambient_temp_c) || !std::isfinite(initial_temp_c)) {
            throw std::invalid_argument("temperatures must be finite");
        }
        if (!(c_th_j_per_c > 0.0) || !std::isfinite(c_th_j_per_c)) {
            throw std::invalid_argument("c_th_j_per_c must be positive");
        }
        if (!(r_th_c_per_w > 0.0) || !std::isfinite(r_th_c_per_w)) {
            throw std::invalid_argument("r_th_c_per_w must be positive");
        }
        if (!(idle_power_w >= 0.0) || !(nominal_power_w > idle_power_w)) {
            throw std::invalid_argument("nominal_power_w must exceed non-negative idle_power_w");
        }
        if (!(nominal_rate_cu_per_s > 0.0) || !std::isfinite(nominal_rate_cu_per_s)) {
            throw std::invalid_argument("nominal_rate_cu_per_s must be positive");
        }
        if (!(severe_throttle_temp_c >= throttle_temp_c)) {
            throw std::invalid_argument("severe_throttle_temp_c must be >= throttle_temp_c");
        }
        if (!(throttle_rate_scale > 0.0 && throttle_rate_scale <= 1.0)) {
            throw std::invalid_argument("throttle_rate_scale must be in (0, 1]");
        }
        if (!(severe_rate_scale > 0.0 && severe_rate_scale <= throttle_rate_scale)) {
            throw std::invalid_argument("severe_rate_scale must be in (0, throttle_rate_scale]");
        }
        if (!(power_exponent >= 1.0) || !std::isfinite(power_exponent)) {
            throw std::invalid_argument("power_exponent must be >= 1.0");
        }
        if (!(throttle_leakage_w >= 0.0) || !std::isfinite(throttle_leakage_w)) {
            throw std::invalid_argument("throttle_leakage_w must be >= 0.0");
        }
        if (!(integration_dt_s > 0.0 && integration_dt_s <= 1.0)) {
            throw std::invalid_argument("integration_dt_s must be in (0, 1]");
        }
    }
};

struct PolicyConfig {
    ExecutionMode mode = ExecutionMode::Burst;
    double fixed_sleep_s = 0.050;
    double bucket_rate_cu_per_s = 60.0;
    double bucket_capacity_cu = 150.0;
    double target_temp_c = 48.0;
    double hysteresis_band_c = 1.5;
    double kp = 0.08;
    double ki = 0.02;
    double min_rate_scale = 0.20;
    double max_rate_scale = 1.00;
    double integral_limit = 25.0;

    void validate() const {
        if (!(fixed_sleep_s >= 0.0) || !std::isfinite(fixed_sleep_s)) {
            throw std::invalid_argument("fixed_sleep_s must be non-negative");
        }
        if (!(bucket_rate_cu_per_s > 0.0) || !std::isfinite(bucket_rate_cu_per_s)) {
            throw std::invalid_argument("bucket_rate_cu_per_s must be positive");
        }
        if (!(bucket_capacity_cu > 0.0) || !std::isfinite(bucket_capacity_cu)) {
            throw std::invalid_argument("bucket_capacity_cu must be positive");
        }
        if (!std::isfinite(target_temp_c) || !(hysteresis_band_c >= 0.0)) {
            throw std::invalid_argument("invalid thermal feedback setpoint or hysteresis");
        }
        if (!(kp >= 0.0) || !(ki >= 0.0) || !std::isfinite(kp) || !std::isfinite(ki)) {
            throw std::invalid_argument("PI controller gains must be non-negative and finite");
        }
        if (!(min_rate_scale > 0.0 && min_rate_scale <= max_rate_scale && max_rate_scale <= 1.0)) {
            throw std::invalid_argument("rate scale bounds must satisfy 0 < min <= max <= 1");
        }
        if (!(integral_limit > 0.0) || !std::isfinite(integral_limit)) {
            throw std::invalid_argument("integral_limit must be positive");
        }
    }
};

struct RequestOutcome {
    std::string request_id;
    double arrival_s = 0.0;
    double start_s = 0.0;
    double finish_s = 0.0;
    double latency_s = 0.0;
    double deadline_s = 0.0;
    bool met_deadline = false;
    double energy_j = 0.0;
    double peak_temp_c = 0.0;
    double throttled_s = 0.0;
};

struct SimulationSummary {
    std::string mode_name;
    std::size_t total_requests = 0;
    std::size_t timely_requests = 0;
    double timely_completion_rate = 0.0;
    double p50_latency_s = 0.0;
    double p95_latency_s = 0.0;
    double p99_latency_s = 0.0;
    double mean_latency_s = 0.0;
    double max_latency_s = 0.0;
    double total_duration_s = 0.0;
    double active_time_s = 0.0;
    double active_duty_factor = 0.0;
    double total_energy_j = 0.0;
    double active_energy_j = 0.0;
    double idle_energy_j = 0.0;
    double peak_temp_c = 0.0;
    double final_temp_c = 0.0;
    double throttled_duration_s = 0.0;
    std::vector<RequestOutcome> outcomes;
};

inline double steady_state_temperature(double power_w, const ThermalModelConfig& cfg) {
    return cfg.ambient_temp_c + power_w * cfg.r_th_c_per_w;
}

inline double advance_rc_temperature(
    double temp_c, double power_w, double dt_s, const ThermalModelConfig& cfg) {
    if (dt_s <= 0.0) return temp_c;
    const double tau_s = cfg.r_th_c_per_w * cfg.c_th_j_per_c;
    const double t_inf = steady_state_temperature(power_w, cfg);
    return t_inf + (temp_c - t_inf) * std::exp(-dt_s / tau_s);
}

inline double hardware_rate_cap(double temp_c, const ThermalModelConfig& cfg, bool* is_throttled) {
    if (temp_c >= cfg.severe_throttle_temp_c) {
        if (is_throttled != nullptr) *is_throttled = true;
        return cfg.severe_rate_scale;
    }
    if (temp_c >= cfg.throttle_temp_c) {
        if (is_throttled != nullptr) *is_throttled = true;
        return cfg.throttle_rate_scale;
    }
    if (is_throttled != nullptr) *is_throttled = false;
    return 1.0;
}

inline double active_power_for_scale(
    double effective_scale, bool hw_throttled, const ThermalModelConfig& cfg) {
    const double clamped = std::clamp(effective_scale, 0.0, 1.0);
    const double dyn_span = cfg.nominal_power_w - cfg.idle_power_w;
    double p_w = cfg.idle_power_w + dyn_span * std::pow(clamped, cfg.power_exponent);
    if (hw_throttled) p_w += cfg.throttle_leakage_w;
    return p_w;
}

inline double compute_percentile(const std::vector<double>& sorted_vals, double q) {
    if (sorted_vals.empty()) return 0.0;
    if (q <= 0.0) return sorted_vals.front();
    if (q >= 1.0) return sorted_vals.back();
    const double pos = q * static_cast<double>(sorted_vals.size() - 1);
    const auto lo = static_cast<std::size_t>(std::floor(pos));
    const auto hi = static_cast<std::size_t>(std::ceil(pos));
    const double frac = pos - static_cast<double>(lo);
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac;
}

class CadenceSimulator {
public:
    CadenceSimulator(ThermalModelConfig thermal_cfg, PolicyConfig policy_cfg)
        : thermal_cfg_(thermal_cfg), policy_cfg_(policy_cfg) {
        thermal_cfg_.validate();
        policy_cfg_.validate();
    }

    SimulationSummary run(const std::vector<RequestSpec>& raw_trace) const {
        SimulationSummary summary;
        summary.mode_name = mode_to_string(policy_cfg_.mode);
        summary.peak_temp_c = thermal_cfg_.initial_temp_c;
        summary.final_temp_c = thermal_cfg_.initial_temp_c;
        if (raw_trace.empty()) return summary;

        std::vector<RequestSpec> trace = raw_trace;
        for (const auto& req : trace) {
            if (req.request_id.empty()) {
                throw std::invalid_argument("request_id must not be empty");
            }
            if (!(req.arrival_s >= 0.0) || !std::isfinite(req.arrival_s)) {
                throw std::invalid_argument("arrival_s must be non-negative and finite");
            }
            if (!(req.compute_units > 0.0) || !std::isfinite(req.compute_units)) {
                throw std::invalid_argument("compute_units must be positive and finite");
            }
            if (!(req.deadline_s > 0.0) || !std::isfinite(req.deadline_s)) {
                throw std::invalid_argument("deadline_s must be positive and finite");
            }
        }
        std::stable_sort(trace.begin(), trace.end(), [](const RequestSpec& a, const RequestSpec& b) {
            return a.arrival_s < b.arrival_s;
        });

        double now_s = 0.0;
        const double sim_origin_s = trace.front().arrival_s;
        double temp_c = thermal_cfg_.initial_temp_c;
        double peak_temp_c = temp_c;
        double total_idle_energy_j = 0.0;
        double total_active_energy_j = 0.0;
        double total_active_time_s = 0.0;
        double total_throttled_s = 0.0;

        double tokens_cu = policy_cfg_.bucket_capacity_cu;
        double pi_integral = 0.0;
        bool feedback_engaged = (temp_c > policy_cfg_.target_temp_c + policy_cfg_.hysteresis_band_c);
        double sw_rate_scale = policy_cfg_.max_rate_scale;

        auto advance_idle = [&](double duration_s) {
            double rem_s = duration_s;
            while (rem_s > 1.0e-15) {
                const double step_s = std::min(rem_s, thermal_cfg_.integration_dt_s);
                temp_c = advance_rc_temperature(temp_c, thermal_cfg_.idle_power_w, step_s, thermal_cfg_);
                peak_temp_c = std::max(peak_temp_c, temp_c);
                total_idle_energy_j += thermal_cfg_.idle_power_w * step_s;
                if (policy_cfg_.mode == ExecutionMode::TokenBucket) {
                    tokens_cu = std::min(
                        policy_cfg_.bucket_capacity_cu,
                        tokens_cu + step_s * policy_cfg_.bucket_rate_cu_per_s);
                } else if (policy_cfg_.mode == ExecutionMode::ThermalFeedback) {
                    update_feedback_scale(temp_c, step_s, feedback_engaged, pi_integral, sw_rate_scale);
                }
                now_s += step_s;
                rem_s -= step_s;
            }
        };

        if (sim_origin_s > 0.0) advance_idle(sim_origin_s);
        summary.outcomes.reserve(trace.size());
        std::vector<double> latencies;
        latencies.reserve(trace.size());

        for (std::size_t idx = 0; idx < trace.size(); ++idx) {
            const RequestSpec& req = trace[idx];
            if (now_s < req.arrival_s) advance_idle(req.arrival_s - now_s);

            if (policy_cfg_.mode == ExecutionMode::TokenBucket) {
                const double needed_tok = std::min(req.compute_units, policy_cfg_.bucket_capacity_cu);
                if (tokens_cu + 1.0e-12 < needed_tok) {
                    advance_idle((needed_tok - tokens_cu) / policy_cfg_.bucket_rate_cu_per_s);
                }
                tokens_cu -= req.compute_units;
            }

            const double start_s = now_s;
            double rem_cu = req.compute_units;
            double req_energy_j = 0.0;
            double req_peak_temp_c = temp_c;
            double req_throttled_s = 0.0;

            while (rem_cu > 1.0e-12) {
                if (policy_cfg_.mode == ExecutionMode::ThermalFeedback) {
                    update_feedback_scale(temp_c, 0.0, feedback_engaged, pi_integral, sw_rate_scale);
                } else {
                    sw_rate_scale = 1.0;
                }
                bool hw_throttled = false;
                const double hw_cap = hardware_rate_cap(temp_c, thermal_cfg_, &hw_throttled);
                const double eff_scale = std::min(sw_rate_scale, hw_cap);
                const double eff_rate = thermal_cfg_.nominal_rate_cu_per_s * eff_scale;
                const double needed_s = rem_cu / eff_rate;
                const double step_s = std::min(needed_s, thermal_cfg_.integration_dt_s);
                const double done_cu = (step_s >= needed_s) ? rem_cu : (eff_rate * step_s);

                const double p_w = active_power_for_scale(eff_scale, hw_throttled, thermal_cfg_);
                temp_c = advance_rc_temperature(temp_c, p_w, step_s, thermal_cfg_);
                peak_temp_c = std::max(peak_temp_c, temp_c);
                req_peak_temp_c = std::max(req_peak_temp_c, temp_c);

                const double d_energy = p_w * step_s;
                req_energy_j += d_energy;
                total_active_energy_j += d_energy;
                total_active_time_s += step_s;
                if (hw_throttled) {
                    req_throttled_s += step_s;
                    total_throttled_s += step_s;
                }
                if (policy_cfg_.mode == ExecutionMode::TokenBucket) {
                    tokens_cu = std::min(
                        policy_cfg_.bucket_capacity_cu,
                        tokens_cu + step_s * policy_cfg_.bucket_rate_cu_per_s);
                } else if (policy_cfg_.mode == ExecutionMode::ThermalFeedback) {
                    update_feedback_scale(temp_c, step_s, feedback_engaged, pi_integral, sw_rate_scale);
                }
                now_s += step_s;
                rem_cu = std::max(0.0, rem_cu - done_cu);
            }

            const double finish_s = now_s;
            const double latency_s = finish_s - req.arrival_s;
            const bool timely = (finish_s <= req.arrival_s + req.deadline_s + 1.0e-12);
            if (timely) ++summary.timely_requests;

            summary.outcomes.push_back({
                req.request_id, req.arrival_s, start_s, finish_s, latency_s,
                req.deadline_s, timely, req_energy_j, req_peak_temp_c, req_throttled_s
            });
            latencies.push_back(latency_s);

            if (policy_cfg_.mode == ExecutionMode::FixedSleep &&
                (idx + 1 < trace.size()) && policy_cfg_.fixed_sleep_s > 0.0) {
                advance_idle(policy_cfg_.fixed_sleep_s);
            }
        }

        summary.total_requests = trace.size();
        summary.timely_completion_rate =
            static_cast<double>(summary.timely_requests) / static_cast<double>(summary.total_requests);

        double sum_lat = 0.0;
        double max_lat = 0.0;
        for (double v : latencies) {
            sum_lat += v;
            max_lat = std::max(max_lat, v);
        }
        std::sort(latencies.begin(), latencies.end());
        summary.p50_latency_s = compute_percentile(latencies, 0.50);
        summary.p95_latency_s = compute_percentile(latencies, 0.95);
        summary.p99_latency_s = compute_percentile(latencies, 0.99);
        summary.mean_latency_s = sum_lat / static_cast<double>(latencies.size());
        summary.max_latency_s = max_lat;

        summary.total_duration_s = now_s;
        summary.active_time_s = total_active_time_s;
        summary.active_duty_factor = (now_s > 0.0) ? (total_active_time_s / now_s) : 0.0;
        summary.active_energy_j = total_active_energy_j;
        summary.idle_energy_j = total_idle_energy_j;
        summary.total_energy_j = total_active_energy_j + total_idle_energy_j;
        summary.peak_temp_c = peak_temp_c;
        summary.final_temp_c = temp_c;
        summary.throttled_duration_s = total_throttled_s;
        return summary;
    }

private:
    ThermalModelConfig thermal_cfg_;
    PolicyConfig policy_cfg_;

    void update_feedback_scale(
        double temp_c, double dt_s, bool& engaged, double& integral_err, double& sw_scale) const {
        const double hi = policy_cfg_.target_temp_c + policy_cfg_.hysteresis_band_c;
        const double lo = policy_cfg_.target_temp_c - policy_cfg_.hysteresis_band_c;
        if (temp_c > hi) {
            engaged = true;
        } else if (temp_c < lo) {
            engaged = false;
            if (dt_s > 0.0) {
                integral_err = std::max(0.0, integral_err + (temp_c - policy_cfg_.target_temp_c) * dt_s);
            }
        }
        const double err = temp_c - policy_cfg_.target_temp_c;
        if (engaged && dt_s > 0.0) {
            integral_err = std::clamp(
                integral_err + err * dt_s, -policy_cfg_.integral_limit, policy_cfg_.integral_limit);
        }
        if (!engaged && integral_err <= 0.0) {
            sw_scale = policy_cfg_.max_rate_scale;
            return;
        }
        const double drop =
            policy_cfg_.kp * std::max(0.0, err) + policy_cfg_.ki * std::max(0.0, integral_err);
        sw_scale = std::clamp(
            policy_cfg_.max_rate_scale - drop, policy_cfg_.min_rate_scale, policy_cfg_.max_rate_scale);
    }
};

}  // namespace cadence

#endif  // CADENCE_CONTROLLER_HPP
