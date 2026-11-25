#ifndef BINOMIAL_BOUNDS_HPP
#define BINOMIAL_BOUNDS_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace calib {

inline double log_beta(double a, double b) {
    if (!(a > 0.0) || !(b > 0.0) || !std::isfinite(a) || !std::isfinite(b)) {
        throw std::domain_error("log_beta requires finite positive parameters");
    }
    return std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
}

inline double betacf(double a, double b, double x) {
    constexpr int kMaxIter = 500;
    constexpr double kEps = 3.0e-15;
    constexpr double kFpMin = 1.0e-300;

    const double qab = a + b;
    const double qap = a + 1.0;
    const double qam = a - 1.0;

    double c = 1.0;
    double d = 1.0 - qab * x / qap;
    if (std::abs(d) < kFpMin) d = kFpMin;
    d = 1.0 / d;
    double h = d;

    for (int m = 1; m <= kMaxIter; ++m) {
        const double dm = static_cast<double>(m);
        const double dm2 = static_cast<double>(2 * m);

        double aa = dm * (b - dm) * x / ((qam + dm2) * (a + dm2));
        d = 1.0 + aa * d;
        if (std::abs(d) < kFpMin) d = kFpMin;
        c = 1.0 + aa / c;
        if (std::abs(c) < kFpMin) c = kFpMin;
        d = 1.0 / d;
        h *= d * c;

        aa = -(a + dm) * (qab + dm) * x / ((a + dm2) * (qap + dm2));
        d = 1.0 + aa * d;
        if (std::abs(d) < kFpMin) d = kFpMin;
        c = 1.0 + aa / c;
        if (std::abs(c) < kFpMin) c = kFpMin;
        d = 1.0 / d;
        const double del = d * c;
        h *= del;

        if (std::abs(del - 1.0) <= kEps) return h;
    }
    throw std::runtime_error("Lentz continued fraction exceeded iteration budget in betacf");
}

inline double betainc(double a, double b, double x) {
    if (!(a > 0.0) || !(b > 0.0) || !std::isfinite(a) || !std::isfinite(b)) {
        throw std::domain_error("betainc requires finite positive shape parameters");
    }
    if (!std::isfinite(x) || x < 0.0 || x > 1.0) {
        throw std::domain_error("betainc requires x in [0, 1]");
    }
    if (x == 0.0) return 0.0;
    if (x == 1.0) return 1.0;

    const double bt = std::exp(a * std::log(x) + b * std::log1p(-x) - log_beta(a, b));
    if (x < (a + 1.0) / (a + b + 2.0)) {
        return std::clamp(bt * betacf(a, b, x) / a, 0.0, 1.0);
    }
    return std::clamp(1.0 - bt * betacf(b, a, 1.0 - x) / b, 0.0, 1.0);
}

inline double beta_inv(double p, double a, double b) {
    if (!(a > 0.0) || !(b > 0.0) || !std::isfinite(a) || !std::isfinite(b)) {
        throw std::domain_error("beta_inv requires finite positive shape parameters");
    }
    if (!std::isfinite(p) || p < 0.0 || p > 1.0) {
        throw std::domain_error("beta_inv requires probability p in [0, 1]");
    }
    if (p == 0.0) return 0.0;
    if (p == 1.0) return 1.0;
    if (a == 1.0 && b == 1.0) return p;
    if (b == 1.0) return std::pow(p, 1.0 / a);
    if (a == 1.0) return -std::expm1(std::log1p(-p) / b);

    double lo = 0.0;
    double hi = 1.0;
    double mid = a / (a + b);
    const double lb = log_beta(a, b);

    for (int iter = 0; iter < 100; ++iter) {
        if (!(mid > lo && mid < hi)) mid = 0.5 * (lo + hi);
        const double f = betainc(a, b, mid) - p;
        if (std::abs(f) <= 1.0e-15 || (hi - lo) <= 1.0e-15 * std::max(1.0, mid)) return mid;
        if (f < 0.0) lo = mid;
        else hi = mid;

        const double pdf = std::exp((a - 1.0) * std::log(mid) + (b - 1.0) * std::log1p(-mid) - lb);
        const double cand = (pdf > 0.0 && std::isfinite(pdf)) ? (mid - f / pdf) : mid;
        mid = (cand > lo && cand < hi) ? cand : 0.5 * (lo + hi);
    }
    return 0.5 * (lo + hi);
}

inline double clopper_pearson_lower(std::size_t k, std::size_t n, double alpha) {
    if (n == 0 || k > n) throw std::domain_error("clopper_pearson_lower requires 0 <= k <= n, n > 0");
    if (!(alpha > 0.0 && alpha < 1.0) || !std::isfinite(alpha)) {
        throw std::domain_error("alpha must lie strictly inside (0, 1)");
    }
    if (k == 0) return 0.0;
    return beta_inv(alpha, static_cast<double>(k), static_cast<double>(n - k + 1));
}

inline double clopper_pearson_upper(std::size_t k, std::size_t n, double alpha) {
    if (n == 0 || k > n) throw std::domain_error("clopper_pearson_upper requires 0 <= k <= n, n > 0");
    if (!(alpha > 0.0 && alpha < 1.0) || !std::isfinite(alpha)) {
        throw std::domain_error("alpha must lie strictly inside (0, 1)");
    }
    if (k == n) return 1.0;
    return beta_inv(1.0 - alpha, static_cast<double>(k + 1), static_cast<double>(n - k));
}

inline std::pair<double, double> clopper_pearson_interval(
    std::size_t k, std::size_t n, double two_sided_alpha) {
    if (!(two_sided_alpha > 0.0 && two_sided_alpha < 1.0) || !std::isfinite(two_sided_alpha)) {
        throw std::domain_error("two_sided_alpha must lie strictly inside (0, 1)");
    }
    return {clopper_pearson_lower(k, n, 0.5 * two_sided_alpha),
            clopper_pearson_upper(k, n, 0.5 * two_sided_alpha)};
}

inline double exact_binomial_upper_tail(std::size_t k, std::size_t n, double p) {
    if (n == 0 || k > n) throw std::domain_error("exact_binomial_upper_tail requires 0 <= k <= n, n > 0");
    if (!(p >= 0.0 && p <= 1.0) || !std::isfinite(p)) throw std::domain_error("p must lie in [0, 1]");
    if (k == 0) return 1.0;
    if (p == 0.0) return 0.0;
    if (p == 1.0) return 1.0;

    double sum = 0.0;
    const double log_n_fact = std::lgamma(static_cast<double>(n + 1));
    for (std::size_t j = k; j <= n; ++j) {
        const double log_comb = log_n_fact - std::lgamma(static_cast<double>(j + 1)) -
                                std::lgamma(static_cast<double>(n - j + 1));
        const double log_pmf = log_comb + static_cast<double>(j) * std::log(p) +
                               static_cast<double>(n - j) * std::log1p(-p);
        sum += std::exp(log_pmf);
    }
    return std::clamp(sum, 0.0, 1.0);
}

enum class UtilityMode { HardDeadline, SoftExponential };
enum class ScreeningMode { ClopperPearsonBonferroni, PointEstimate };

struct EvaluationRecord {
    std::string request_id;
    std::string config_id;
    bool passed = false;
    double latency_s = 0.0;
    double energy_j = 0.0;
    std::size_t tokens = 0;
};

struct CalibrationParams {
    UtilityMode utility_mode = UtilityMode::HardDeadline;
    ScreeningMode screening_mode = ScreeningMode::ClopperPearsonBonferroni;
    double deadline_s = 1.0;
    double tau_s = 1.0;
    double target_utility = 0.90;
    double delta = 0.05;

    void validate() const {
        if (utility_mode == UtilityMode::HardDeadline) {
            if (!(deadline_s > 0.0) || !std::isfinite(deadline_s)) {
                throw std::invalid_argument("deadline_s must be positive for hard_deadline");
            }
        } else if (!(tau_s > 0.0) || !std::isfinite(tau_s)) {
            throw std::invalid_argument("tau_s must be positive for soft_exp");
        }
        if (!(target_utility > 0.0 && target_utility <= 1.0) || !std::isfinite(target_utility)) {
            throw std::invalid_argument("target_utility must be in (0, 1]");
        }
        if (!(delta > 0.0 && delta < 1.0) || !std::isfinite(delta)) {
            throw std::invalid_argument("delta must be in (0, 1)");
        }
    }
};

struct CandidateStats {
    std::string config_id;
    std::size_t sample_count = 0;
    double utility_sum = 0.0;
    std::size_t effective_successes = 0;
    double empirical_utility = 0.0;
    double cp_lower_bound = 0.0;
    double cp_upper_bound = 1.0;
    double screening_score = 0.0;
    bool cal_feasible = false;
    double mean_energy_j = 0.0;
    double mean_latency_s = 0.0;
    double energy_per_token_j = 0.0;

    bool has_eval = false;
    std::size_t eval_sample_count = 0;
    double eval_utility = 0.0;
    double eval_cp_lower_bound = 0.0;
    double eval_mean_energy_j = 0.0;
    double eval_mean_latency_s = 0.0;
    bool eval_attains_target = false;
};

struct CalibrationReport {
    std::size_t num_candidates = 0;
    double bonferroni_alpha = 0.0;
    bool feasible = false;
    std::string selected_config_id;
    double selected_cal_energy_j = 0.0;
    double selected_cal_utility = 0.0;
    double selected_cal_certificate = 0.0;

    bool has_eval = false;
    bool held_out_target_attained = false;
    double held_out_utility = 0.0;
    double held_out_mean_energy_j = 0.0;

    bool oracle_feasible = false;
    std::string oracle_config_id;
    double oracle_mean_energy_j = 0.0;
    double oracle_utility = 0.0;
    double oracle_energy_gap_j = 0.0;
    double oracle_energy_ratio = 0.0;

    std::vector<CandidateStats> candidates;
};

inline double record_utility(const EvaluationRecord& rec, const CalibrationParams& params) {
    if (!(rec.latency_s >= 0.0) || !std::isfinite(rec.latency_s)) {
        throw std::invalid_argument("record latency_s must be non-negative and finite");
    }
    if (!(rec.energy_j >= 0.0) || !std::isfinite(rec.energy_j)) {
        throw std::invalid_argument("record energy_j must be non-negative and finite");
    }
    if (!rec.passed) return 0.0;
    if (params.utility_mode == UtilityMode::HardDeadline) {
        return (rec.latency_s <= params.deadline_s) ? 1.0 : 0.0;
    }
    return std::exp(-rec.latency_s / params.tau_s);
}

inline CalibrationReport calibrate_and_evaluate(
    const std::vector<EvaluationRecord>& cal_records,
    const std::vector<EvaluationRecord>& eval_records,
    const CalibrationParams& params) {
    params.validate();
    if (cal_records.empty()) throw std::invalid_argument("calibration dataset must not be empty");

    struct Accum {
        std::size_t n = 0;
        double u_sum = 0.0, e_sum = 0.0, l_sum = 0.0;
        std::size_t tok_sum = 0;
    };

    auto accumulate_set = [&](const std::vector<EvaluationRecord>& recs) {
        std::map<std::string, Accum> m;
        for (const auto& rec : recs) {
            if (rec.config_id.empty() || rec.request_id.empty()) {
                throw std::invalid_argument("config_id and request_id must be non-empty");
            }
            const double u = record_utility(rec, params);
            Accum& acc = m[rec.config_id];
            ++acc.n;
            acc.u_sum += u;
            acc.e_sum += rec.energy_j;
            acc.l_sum += rec.latency_s;
            acc.tok_sum += rec.tokens;
        }
        return m;
    };

    const auto cal_map = accumulate_set(cal_records);
    const auto eval_map = accumulate_set(eval_records);

    CalibrationReport report;
    report.num_candidates = cal_map.size();
    report.bonferroni_alpha = params.delta / static_cast<double>(report.num_candidates);
    report.has_eval = !eval_records.empty();
    report.candidates.reserve(cal_map.size());

    auto better_energy = [](double e_a, const std::string& id_a, double e_b, const std::string& id_b) {
        return (std::abs(e_a - e_b) > 1.0e-12) ? (e_a < e_b) : (id_a < id_b);
    };

    for (const auto& kv : cal_map) {
        const std::string& cid = kv.first;
        const Accum& acc = kv.second;

        CandidateStats st;
        st.config_id = cid;
        st.sample_count = acc.n;
        st.utility_sum = acc.u_sum;
        st.effective_successes = std::min(
            acc.n, static_cast<std::size_t>(std::floor(acc.u_sum + 1.0e-12)));
        st.empirical_utility = acc.u_sum / static_cast<double>(acc.n);
        st.cp_lower_bound = clopper_pearson_lower(
            st.effective_successes, st.sample_count, report.bonferroni_alpha);
        const std::size_t ceil_succ = std::min(
            acc.n, static_cast<std::size_t>(std::ceil(acc.u_sum - 1.0e-12)));
        st.cp_upper_bound = clopper_pearson_upper(
            ceil_succ, st.sample_count, report.bonferroni_alpha);
        st.screening_score = (params.screening_mode == ScreeningMode::ClopperPearsonBonferroni)
                                 ? st.cp_lower_bound
                                 : st.empirical_utility;
        st.cal_feasible = (st.screening_score + 1.0e-15 >= params.target_utility);
        st.mean_energy_j = acc.e_sum / static_cast<double>(acc.n);
        st.mean_latency_s = acc.l_sum / static_cast<double>(acc.n);
        st.energy_per_token_j = (acc.tok_sum > 0) ? (acc.e_sum / static_cast<double>(acc.tok_sum)) : 0.0;

        const auto eval_it = eval_map.find(cid);
        if (eval_it != eval_map.end() && eval_it->second.n > 0) {
            const Accum& eacc = eval_it->second;
            st.has_eval = true;
            st.eval_sample_count = eacc.n;
            st.eval_utility = eacc.u_sum / static_cast<double>(eacc.n);
            const std::size_t eval_succ = std::min(
                eacc.n, static_cast<std::size_t>(std::floor(eacc.u_sum + 1.0e-12)));
            st.eval_cp_lower_bound = clopper_pearson_lower(eval_succ, eacc.n, params.delta);
            st.eval_mean_energy_j = eacc.e_sum / static_cast<double>(eacc.n);
            st.eval_mean_latency_s = eacc.l_sum / static_cast<double>(eacc.n);
            st.eval_attains_target = (st.eval_utility + 1.0e-15 >= params.target_utility);
        }
        report.candidates.push_back(st);
    }

    const CandidateStats* best_cal = nullptr;
    const CandidateStats* best_oracle = nullptr;
    for (const auto& st : report.candidates) {
        if (st.cal_feasible && (best_cal == nullptr ||
            better_energy(st.mean_energy_j, st.config_id, best_cal->mean_energy_j, best_cal->config_id))) {
            best_cal = &st;
        }
        if (st.has_eval && st.eval_attains_target && (best_oracle == nullptr ||
            better_energy(st.eval_mean_energy_j, st.config_id, best_oracle->eval_mean_energy_j, best_oracle->config_id))) {
            best_oracle = &st;
        }
    }

    if (best_cal != nullptr) {
        report.feasible = true;
        report.selected_config_id = best_cal->config_id;
        report.selected_cal_energy_j = best_cal->mean_energy_j;
        report.selected_cal_utility = best_cal->empirical_utility;
        report.selected_cal_certificate = best_cal->screening_score;
        if (best_cal->has_eval) {
            report.held_out_utility = best_cal->eval_utility;
            report.held_out_mean_energy_j = best_cal->eval_mean_energy_j;
            report.held_out_target_attained = best_cal->eval_attains_target;
        }
    }
    if (best_oracle != nullptr) {
        report.oracle_feasible = true;
        report.oracle_config_id = best_oracle->config_id;
        report.oracle_mean_energy_j = best_oracle->eval_mean_energy_j;
        report.oracle_utility = best_oracle->eval_utility;
        if (report.feasible && best_cal->has_eval) {
            report.oracle_energy_gap_j = report.held_out_mean_energy_j - report.oracle_mean_energy_j;
            report.oracle_energy_ratio = (report.oracle_mean_energy_j > 0.0)
                                             ? (report.held_out_mean_energy_j / report.oracle_mean_energy_j)
                                             : 1.0;
        }
    }
    return report;
}

}  // namespace calib

#endif  // BINOMIAL_BOUNDS_HPP
