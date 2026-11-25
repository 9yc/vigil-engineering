#ifndef WAVE_TASK_DAG_HPP_
#define WAVE_TASK_DAG_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace task_dag {

struct TaskSpec {
  std::string id;
  double cost = 0.0;
  std::vector<std::string> deps;
};

struct ValidationResult {
  bool valid = false;
  std::string code;
  std::string message;
  std::vector<std::string> witness_cycle;
};

struct NodeCpmMetrics {
  std::string id;
  double cost = 0.0;
  double earliest_start = 0.0;
  double earliest_finish = 0.0;
  double latest_start = 0.0;
  double latest_finish = 0.0;
  double total_slack = 0.0;
  double bottom_level = 0.0;
  bool critical = false;
};

struct CpmReport {
  std::vector<std::string> topological_order;
  double total_work = 0.0;
  double critical_path_cost = 0.0;
  std::vector<std::string> critical_path;
  std::vector<NodeCpmMetrics> nodes;
};

struct ScheduledTask {
  std::string task_id;
  uint32_t worker_id = 0;
  double start_time = 0.0;
  double finish_time = 0.0;
};

struct ScheduleResult {
  uint32_t workers = 1;
  double makespan = 0.0;
  double lower_bound = 0.0;
  bool bounds_satisfied = true;
  double total_work = 0.0;
  double average_utilization = 0.0;
  double speedup_vs_serial = 1.0;
  std::vector<double> worker_busy_time;
  std::vector<double> worker_utilization;
  std::vector<ScheduledTask> timeline;
};

class TaskDag {
 public:
  static constexpr double kEps = 1e-9;

  explicit TaskDag(std::vector<TaskSpec> specs) : specs_(std::move(specs)) {}

  ValidationResult Validate() {
    id_to_idx_.clear();
    adj_.clear();
    preds_.clear();
    indegree_.clear();
    topo_indices_.clear();

    const size_t n = specs_.size();
    for (size_t i = 0; i < n; ++i) {
      const TaskSpec& t = specs_[i];
      if (t.id.empty()) return {false, "EMPTY_ID", "Task ID must be non-empty", {}};
      if (!std::isfinite(t.cost) || t.cost < 0.0) {
        return {false, "INVALID_COST", "Negative or non-finite task cost: " + t.id, {}};
      }
      if (!id_to_idx_.emplace(t.id, i).second) {
        return {false, "DUPLICATE_ID", "Duplicate task ID: " + t.id, {}};
      }
    }

    adj_.assign(n, {});
    preds_.assign(n, {});
    indegree_.assign(n, 0);
    for (size_t i = 0; i < n; ++i) {
      std::set<size_t> uniq;
      for (const std::string& dep : specs_[i].deps) {
        auto it = id_to_idx_.find(dep);
        if (it == id_to_idx_.end()) {
          return {false, "MISSING_DEPENDENCY",
                  "Task '" + specs_[i].id + "' depends on missing '" + dep + "'", {}};
        }
        uniq.insert(it->second);
      }
      for (size_t d : uniq) {
        adj_[d].push_back(i);
        preds_[i].push_back(d);
        ++indegree_[i];
      }
    }
    for (size_t i = 0; i < n; ++i) {
      auto by_name = [&](size_t a, size_t b) { return specs_[a].id < specs_[b].id; };
      std::sort(adj_[i].begin(), adj_[i].end(), by_name);
      std::sort(preds_[i].begin(), preds_[i].end(), by_name);
    }

    auto min_id = [&](size_t a, size_t b) { return specs_[a].id > specs_[b].id; };
    std::priority_queue<size_t, std::vector<size_t>, decltype(min_id)> ready(min_id);
    std::vector<size_t> deg = indegree_;
    for (size_t i = 0; i < n; ++i) {
      if (deg[i] == 0) ready.push(i);
    }
    std::vector<bool> visited(n, false);
    while (!ready.empty()) {
      size_t u = ready.top();
      ready.pop();
      visited[u] = true;
      topo_indices_.push_back(u);
      for (size_t v : adj_[u]) {
        if (--deg[v] == 0) ready.push(v);
      }
    }

    if (topo_indices_.size() != n) {
      std::vector<std::string> cyc = ExtractMinimalCycle(visited);
      std::string msg = "Cycle detected: ";
      for (size_t i = 0; i < cyc.size(); ++i) msg += (i ? " -> " : "") + cyc[i];
      return {false, "CYCLE_DETECTED", msg, std::move(cyc)};
    }
    validated_ = true;
    return {true, "OK", "Valid DAG", {}};
  }

  CpmReport AnalyzeCpm() {
    EnsureValidated();
    const size_t n = specs_.size();
    CpmReport rep;
    if (n == 0) return rep;

    std::vector<double> es(n, 0.0), ef(n, 0.0);
    std::vector<int64_t> crit_pred(n, -1);
    double total_work = 0.0, cp_cost = 0.0;
    for (size_t u : topo_indices_) {
      total_work += specs_[u].cost;
      double best_es = 0.0;
      int64_t best_p = -1;
      for (size_t p : preds_[u]) {
        if (ef[p] > best_es + kEps ||
            (std::abs(ef[p] - best_es) <= kEps &&
             (best_p < 0 || specs_[p].id < specs_[static_cast<size_t>(best_p)].id))) {
          best_es = ef[p];
          best_p = static_cast<int64_t>(p);
        }
      }
      es[u] = best_es;
      ef[u] = best_es + specs_[u].cost;
      crit_pred[u] = best_p;
      cp_cost = std::max(cp_cost, ef[u]);
    }

    std::vector<double> lf(n, cp_cost), ls(n, 0.0), bl(n, 0.0);
    for (auto it = topo_indices_.rbegin(); it != topo_indices_.rend(); ++it) {
      size_t u = *it;
      if (adj_[u].empty()) {
        lf[u] = cp_cost;
        bl[u] = specs_[u].cost;
      } else {
        double min_ls = std::numeric_limits<double>::infinity(), max_bl = 0.0;
        for (size_t v : adj_[u]) {
          min_ls = std::min(min_ls, ls[v]);
          max_bl = std::max(max_bl, bl[v]);
        }
        lf[u] = min_ls;
        bl[u] = specs_[u].cost + max_bl;
      }
      ls[u] = lf[u] - specs_[u].cost;
    }

    size_t sink = topo_indices_.front();
    bool has_sink = false;
    for (size_t u : topo_indices_) {
      if (std::abs(ef[u] - cp_cost) <= kEps && (!has_sink || specs_[u].id < specs_[sink].id)) {
        sink = u;
        has_sink = true;
      }
    }
    if (has_sink) {
      std::vector<size_t> chain;
      for (int64_t c = static_cast<int64_t>(sink); c >= 0; c = crit_pred[static_cast<size_t>(c)]) {
        chain.push_back(static_cast<size_t>(c));
      }
      for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
        rep.critical_path.push_back(specs_[*it].id);
      }
    }

    rep.total_work = total_work_ = total_work;
    rep.critical_path_cost = critical_path_cost_ = cp_cost;
    for (size_t u : topo_indices_) rep.topological_order.push_back(specs_[u].id);
    bottom_level_ = bl;
    earliest_start_ = es;
    cpm_ready_ = true;

    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) { return specs_[a].id < specs_[b].id; });
    for (size_t i : order) {
      double slack = (std::abs(ls[i] - es[i]) <= kEps) ? 0.0 : (ls[i] - es[i]);
      rep.nodes.push_back({specs_[i].id, specs_[i].cost, es[i], ef[i], ls[i], lf[i],
                           slack, bl[i], slack <= kEps});
    }
    return rep;
  }

  ScheduleResult SimulateWorkers(uint32_t workers) {
    if (workers == 0) throw std::invalid_argument("Workers must be >= 1");
    if (!cpm_ready_) AnalyzeCpm();
    const size_t n = specs_.size();
    ScheduleResult res;
    res.workers = workers;
    res.total_work = total_work_;
    res.lower_bound = std::max(critical_path_cost_, total_work_ / static_cast<double>(workers));
    res.worker_busy_time.assign(workers, 0.0);
    res.worker_utilization.assign(workers, 0.0);
    if (n == 0) {
      res.average_utilization = 1.0;
      return res;
    }

    auto prio = [&](size_t a, size_t b) {
      if (std::abs(bottom_level_[a] - bottom_level_[b]) > kEps) {
        return bottom_level_[a] < bottom_level_[b];
      }
      if (std::abs(earliest_start_[a] - earliest_start_[b]) > kEps) {
        return earliest_start_[a] > earliest_start_[b];
      }
      return specs_[a].id > specs_[b].id;
    };
    std::priority_queue<size_t, std::vector<size_t>, decltype(prio)> ready(prio);

    struct Ev {
      double fin;
      uint32_t worker;
      size_t idx;
      bool operator>(const Ev& o) const noexcept {
        return (fin != o.fin) ? (fin > o.fin) : (worker > o.worker);
      }
    };
    std::priority_queue<Ev, std::vector<Ev>, std::greater<>> active;
    std::priority_queue<uint32_t, std::vector<uint32_t>, std::greater<>> idle;
    for (uint32_t w = 0; w < workers; ++w) idle.push(w);

    std::vector<size_t> rem = indegree_;
    for (size_t i = 0; i < n; ++i) {
      if (rem[i] == 0) ready.push(i);
    }

    double now = 0.0;
    size_t done = 0;
    while (done < n) {
      while (!ready.empty() && !idle.empty()) {
        size_t u = ready.top();
        ready.pop();
        uint32_t w = idle.top();
        if (specs_[u].cost <= 0.0) {
          res.timeline.push_back({specs_[u].id, w, now, now});
          ++done;
          for (size_t v : adj_[u]) {
            if (--rem[v] == 0) ready.push(v);
          }
        } else {
          idle.pop();
          double fin = now + specs_[u].cost;
          res.worker_busy_time[w] += specs_[u].cost;
          res.timeline.push_back({specs_[u].id, w, now, fin});
          active.push({fin, w, u});
        }
      }
      if (done == n) break;
      now = active.top().fin;
      while (!active.empty() && std::abs(active.top().fin - now) <= kEps) {
        Ev ev = active.top();
        active.pop();
        idle.push(ev.worker);
        ++done;
        for (size_t v : adj_[ev.idx]) {
          if (--rem[v] == 0) ready.push(v);
        }
      }
    }

    res.makespan = now;
    res.bounds_satisfied = (res.makespan + kEps >= res.lower_bound);
    if (res.makespan > kEps) {
      res.average_utilization = res.total_work / (res.makespan * static_cast<double>(workers));
      res.speedup_vs_serial = res.total_work / res.makespan;
      for (uint32_t w = 0; w < workers; ++w) {
        res.worker_utilization[w] = res.worker_busy_time[w] / res.makespan;
      }
    } else {
      res.average_utilization = 1.0;
    }
    return res;
  }

 private:
  void EnsureValidated() {
    if (!validated_) {
      ValidationResult vr = Validate();
      if (!vr.valid) throw std::runtime_error(vr.code + ": " + vr.message);
    }
  }

  std::vector<std::string> ExtractMinimalCycle(const std::vector<bool>& kahn_vis) const {
    const size_t n = specs_.size();
    std::vector<size_t> cands;
    for (size_t i = 0; i < n; ++i) {
      if (!kahn_vis[i]) cands.push_back(i);
    }
    std::sort(cands.begin(), cands.end(),
              [&](size_t a, size_t b) { return specs_[a].id < specs_[b].id; });

    std::vector<std::string> best;
    for (size_t start : cands) {
      std::vector<int64_t> parent(n, -1), dist(n, -1);
      std::queue<size_t> q;
      for (size_t nxt : adj_[start]) {
        if (kahn_vis[nxt]) continue;
        dist[nxt] = 1;
        parent[nxt] = static_cast<int64_t>(start);
        q.push(nxt);
      }
      while (!q.empty() && dist[start] < 0) {
        size_t u = q.front();
        q.pop();
        if (!best.empty() && static_cast<size_t>(dist[u] + 1) >= best.size()) continue;
        for (size_t v : adj_[u]) {
          if (kahn_vis[v] || dist[v] >= 0) continue;
          dist[v] = dist[u] + 1;
          parent[v] = static_cast<int64_t>(u);
          if (v == start) break;
          q.push(v);
        }
      }
      if (dist[start] > 0) {
        std::vector<size_t> p{start};
        for (size_t c = static_cast<size_t>(parent[start]); c != start;
             c = static_cast<size_t>(parent[c])) {
          p.push_back(c);
        }
        p.push_back(start);
        std::reverse(p.begin(), p.end());
        std::vector<std::string> names;
        for (size_t idx : p) names.push_back(specs_[idx].id);
        if (best.empty() || names.size() < best.size() ||
            (names.size() == best.size() && names < best)) {
          best = std::move(names);
        }
      }
    }
    return best;
  }

  std::vector<TaskSpec> specs_;
  std::unordered_map<std::string, size_t> id_to_idx_;
  std::vector<std::vector<size_t>> adj_, preds_;
  std::vector<size_t> indegree_, topo_indices_;
  bool validated_ = false, cpm_ready_ = false;
  std::vector<double> bottom_level_, earliest_start_;
  double critical_path_cost_ = 0.0, total_work_ = 0.0;
};

}  // namespace task_dag

#endif  // WAVE_TASK_DAG_HPP_
