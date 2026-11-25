#ifndef WAVE_SLAB_ARENA_HPP_
#define WAVE_SLAB_ARENA_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace slab_arena {

enum class Strategy { kFreeListFirstFit, kFreeListBestFit, kBuddy };
enum class FreeStatus { kOk, kDoubleFree, kInvalidHandle };

struct Allocation {
  uint64_t handle = 0, raw_offset = 0, payload_offset = 0;
  uint64_t payload_size = 0, reserved_size = 0, alignment = 1, alignment_padding = 0;
};

struct ArenaStats {
  uint64_t capacity_bytes = 0, accepted = 0, failed = 0, freed = 0;
  uint64_t invalid_frees = 0, double_frees = 0, live_allocations = 0;
  uint64_t live_reserved_bytes = 0, live_payload_bytes = 0;
  uint64_t total_free_bytes = 0, largest_free_bytes = 0;
  uint64_t peak_reserved_bytes = 0, peak_payload_bytes = 0;
  uint64_t alignment_overhead_bytes = 0, live_alignment_overhead_bytes = 0;
  uint64_t peak_alignment_overhead_bytes = 0;
  double external_fragmentation = 0.0, peak_external_fragmentation = 0.0;
};

class SlabArena {
 public:
  static constexpr uint32_t kNullIdx = std::numeric_limits<uint32_t>::max();
  static constexpr size_t kNumBins = 64;

  SlabArena(uint64_t capacity_bytes, Strategy strategy, uint64_t min_block_bytes = 16)
      : capacity_bytes_(capacity_bytes), strategy_(strategy), min_block_bytes_(min_block_bytes) {
    if (capacity_bytes_ == 0) throw std::invalid_argument("Arena capacity must be positive");
    if (strategy_ == Strategy::kBuddy) {
      if (!IsPowerOfTwo(capacity_bytes_) || !IsPowerOfTwo(min_block_bytes_) ||
          min_block_bytes_ > capacity_bytes_) {
        throw std::invalid_argument("Buddy strategy requires power-of-two geometry");
      }
      max_order_ = FloorLog2(capacity_bytes_ / min_block_bytes_);
      buddy_free_.resize(max_order_ + 1);
      buddy_free_[max_order_].insert(0);
      buddy_free_mask_ = uint64_t{1} << max_order_;
    } else {
      head_idx_ = tail_idx_ = CreateSpanNode(0, capacity_bytes_, true);
      InsertFreeSpan(head_idx_);
    }
    RefreshFragmentation();
  }

  static constexpr bool IsPowerOfTwo(uint64_t v) noexcept { return v != 0 && (v & (v - 1)) == 0; }
  static constexpr size_t FloorLog2(uint64_t v) noexcept {
    size_t r = 0;
    while (v > 1) { v >>= 1; ++r; }
    return r;
  }

  static bool NextPowerOfTwo(uint64_t v, uint64_t* out) noexcept {
    if (v == 0) { *out = 1; return true; }
    if (v > (uint64_t{1} << 63)) return false;
    uint64_t p = 1;
    while (p < v) p <<= 1;
    *out = p;
    return true;
  }

  static bool AlignUp(uint64_t offset, uint64_t alignment, uint64_t* aligned) noexcept {
    const uint64_t mask = alignment - 1;
    if (offset > std::numeric_limits<uint64_t>::max() - mask) return false;
    *aligned = (offset + mask) & ~mask;
    return true;
  }

  std::optional<Allocation> Allocate(uint64_t size, uint64_t alignment = 1) {
    if (size == 0 || !IsPowerOfTwo(alignment) || size > capacity_bytes_) {
      ++failed_;
      return std::nullopt;
    }
    auto res = (strategy_ == Strategy::kBuddy) ? AllocateBuddy(size, alignment)
                                               : AllocateFreeList(size, alignment);
    if (!res.has_value()) { ++failed_; return std::nullopt; }
    ++accepted_;
    live_reserved_bytes_ += res->reserved_size;
    live_payload_bytes_ += res->payload_size;
    alignment_overhead_bytes_ += res->alignment_padding;
    live_alignment_overhead_bytes_ += res->alignment_padding;
    peak_reserved_bytes_ = std::max(peak_reserved_bytes_, live_reserved_bytes_);
    peak_payload_bytes_ = std::max(peak_payload_bytes_, live_payload_bytes_);
    peak_alignment_overhead_bytes_ =
        std::max(peak_alignment_overhead_bytes_, live_alignment_overhead_bytes_);
    RefreshFragmentation();
    return res;
  }

  FreeStatus Deallocate(uint64_t handle) {
    if (handle == 0 || handle >= next_handle_) { ++invalid_frees_; return FreeStatus::kInvalidHandle; }
    if (retired_handles_.count(handle) != 0) {
      ++invalid_frees_;
      ++double_frees_;
      return FreeStatus::kDoubleFree;
    }
    Allocation removed{};
    if (strategy_ == Strategy::kBuddy) {
      auto it = buddy_live_.find(handle);
      if (it == buddy_live_.end()) { ++invalid_frees_; return FreeStatus::kInvalidHandle; }
      removed = it->second;
      buddy_live_.erase(it);
      DeallocateBuddyBlock(removed.raw_offset, removed.reserved_size);
    } else {
      auto it = freelist_live_.find(handle);
      if (it == freelist_live_.end()) { ++invalid_frees_; return FreeStatus::kInvalidHandle; }
      uint32_t idx = it->second;
      freelist_live_.erase(it);
      removed = nodes_[idx].alloc;
      nodes_[idx].is_free = true;
      nodes_[idx].alloc = Allocation{};
      CoalesceAndInsert(idx);
    }
    retired_handles_.insert(handle);
    ++freed_;
    live_reserved_bytes_ -= removed.reserved_size;
    live_payload_bytes_ -= removed.payload_size;
    live_alignment_overhead_bytes_ -= removed.alignment_padding;
    RefreshFragmentation();
    return FreeStatus::kOk;
  }

  ArenaStats Stats() const noexcept {
    const uint64_t live_cnt =
        (strategy_ == Strategy::kBuddy) ? buddy_live_.size() : freelist_live_.size();
    return ArenaStats{capacity_bytes_, accepted_, failed_, freed_, invalid_frees_, double_frees_,
                      live_cnt, live_reserved_bytes_, live_payload_bytes_,
                      capacity_bytes_ - live_reserved_bytes_, LargestFreeBlock(),
                      peak_reserved_bytes_, peak_payload_bytes_, alignment_overhead_bytes_,
                      live_alignment_overhead_bytes_, peak_alignment_overhead_bytes_,
                      current_external_fragmentation_, peak_external_fragmentation_};
  }

  bool VerifyInvariants(std::string* err = nullptr) const {
    return (strategy_ == Strategy::kBuddy) ? VerifyBuddyInvariants(err)
                                           : VerifyFreeListInvariants(err);
  }

 private:
  struct SpanNode {
    uint64_t offset = 0, length = 0;
    uint32_t prev_phys = kNullIdx, next_phys = kNullIdx;
    bool is_free = true, in_use = true;
    Allocation alloc{};
  };
  struct BinEntry {
    uint64_t primary = 0, secondary = 0;
    uint32_t idx = kNullIdx;
    bool operator<(const BinEntry& o) const noexcept {
      if (primary != o.primary) return primary < o.primary;
      return (secondary != o.secondary) ? (secondary < o.secondary) : (idx < o.idx);
    }
  };

  uint32_t CreateSpanNode(uint64_t offset, uint64_t length, bool is_free) {
    if (!recycled_nodes_.empty()) {
      uint32_t idx = recycled_nodes_.back();
      recycled_nodes_.pop_back();
      nodes_[idx] = SpanNode{offset, length, kNullIdx, kNullIdx, is_free, true, {}};
      return idx;
    }
    nodes_.push_back(SpanNode{offset, length, kNullIdx, kNullIdx, is_free, true, {}});
    return static_cast<uint32_t>(nodes_.size() - 1);
  }
  void RecycleSpanNode(uint32_t idx) {
    nodes_[idx].in_use = false;
    nodes_[idx].prev_phys = nodes_[idx].next_phys = kNullIdx;
    recycled_nodes_.push_back(idx);
  }
  BinEntry MakeBinEntry(uint32_t idx) const noexcept {
    const SpanNode& n = nodes_[idx];
    return (strategy_ == Strategy::kFreeListFirstFit) ? BinEntry{n.offset, n.length, idx}
                                                      : BinEntry{n.length, n.offset, idx};
  }
  void InsertFreeSpan(uint32_t idx) {
    const size_t bin = FloorLog2(nodes_[idx].length);
    bins_[bin].insert(MakeBinEntry(idx));
    non_empty_bins_ |= (uint64_t{1} << bin);
  }
  void RemoveFreeSpan(uint32_t idx) {
    const size_t bin = FloorLog2(nodes_[idx].length);
    bins_[bin].erase(MakeBinEntry(idx));
    if (bins_[bin].empty()) non_empty_bins_ &= ~(uint64_t{1} << bin);
  }

  static bool EvalSpanFit(const SpanNode& n, uint64_t size, uint64_t alignment,
                          uint64_t* aligned_out, uint64_t* needed_out) noexcept {
    uint64_t aligned = 0;
    if (!AlignUp(n.offset, alignment, &aligned) || aligned < n.offset) return false;
    const uint64_t pad = aligned - n.offset;
    if (pad > n.length || size > n.length - pad) return false;
    *aligned_out = aligned;
    *needed_out = pad + size;
    return true;
  }

  std::optional<Allocation> AllocateFreeList(uint64_t size, uint64_t alignment) {
    const size_t min_bin = FloorLog2(size);
    if (min_bin >= kNumBins || (non_empty_bins_ >> min_bin) == 0) return std::nullopt;
    uint32_t chosen_idx = kNullIdx;
    uint64_t chosen_aligned = 0, chosen_needed = 0;

    if (strategy_ == Strategy::kFreeListFirstFit) {
      uint64_t best_off = std::numeric_limits<uint64_t>::max();
      for (size_t b = min_bin; b < kNumBins; ++b) {
        if (((non_empty_bins_ >> b) & 1ULL) == 0) continue;
        for (const BinEntry& e : bins_[b]) {
          if (e.primary >= best_off) break;
          uint64_t al = 0, nd = 0;
          if (EvalSpanFit(nodes_[e.idx], size, alignment, &al, &nd)) {
            best_off = e.primary; chosen_idx = e.idx; chosen_aligned = al; chosen_needed = nd;
            break;
          }
        }
      }
    } else {
      uint64_t best_slack = std::numeric_limits<uint64_t>::max();
      uint64_t best_off = std::numeric_limits<uint64_t>::max();
      for (size_t b = min_bin; b < kNumBins; ++b) {
        if (((non_empty_bins_ >> b) & 1ULL) == 0) continue;
        const uint64_t min_len = uint64_t{1} << b;
        if (chosen_idx != kNullIdx && min_len > size && (min_len - size) > best_slack &&
            (best_slack == 0 || min_len - size - (alignment - 1) > best_slack)) break;
        for (const BinEntry& e : bins_[b]) {
          uint64_t al = 0, nd = 0;
          if (!EvalSpanFit(nodes_[e.idx], size, alignment, &al, &nd)) continue;
          const uint64_t slack = nodes_[e.idx].length - nd;
          if (slack < best_slack || (slack == best_slack && nodes_[e.idx].offset < best_off)) {
            best_slack = slack; best_off = nodes_[e.idx].offset;
            chosen_idx = e.idx; chosen_aligned = al; chosen_needed = nd;
            if (alignment == 1) break;
          }
        }
        if (alignment == 1 && chosen_idx != kNullIdx) break;
      }
    }
    if (chosen_idx == kNullIdx) return std::nullopt;

    RemoveFreeSpan(chosen_idx);
    const uint64_t orig_off = nodes_[chosen_idx].offset, orig_len = nodes_[chosen_idx].length;
    const uint32_t orig_next = nodes_[chosen_idx].next_phys;
    if (orig_len > chosen_needed) {
      uint32_t rem = CreateSpanNode(orig_off + chosen_needed, orig_len - chosen_needed, true);
      nodes_[chosen_idx].length = chosen_needed;
      nodes_[chosen_idx].next_phys = rem;
      nodes_[rem].prev_phys = chosen_idx;
      nodes_[rem].next_phys = orig_next;
      if (orig_next != kNullIdx) nodes_[orig_next].prev_phys = rem;
      else tail_idx_ = rem;
      InsertFreeSpan(rem);
    }
    Allocation alloc{next_handle_++, orig_off, chosen_aligned, size,
                     chosen_needed, alignment, chosen_aligned - orig_off};
    nodes_[chosen_idx].is_free = false;
    nodes_[chosen_idx].alloc = alloc;
    freelist_live_.emplace(alloc.handle, chosen_idx);
    return alloc;
  }

  void CoalesceAndInsert(uint32_t idx) {
    uint32_t prev = nodes_[idx].prev_phys;
    if (prev != kNullIdx && nodes_[prev].is_free) {
      RemoveFreeSpan(prev);
      nodes_[prev].length += nodes_[idx].length;
      uint32_t next = nodes_[idx].next_phys;
      nodes_[prev].next_phys = next;
      if (next != kNullIdx) nodes_[next].prev_phys = prev;
      else tail_idx_ = prev;
      RecycleSpanNode(idx);
      idx = prev;
    }
    uint32_t next = nodes_[idx].next_phys;
    if (next != kNullIdx && nodes_[next].is_free) {
      RemoveFreeSpan(next);
      nodes_[idx].length += nodes_[next].length;
      uint32_t nn = nodes_[next].next_phys;
      nodes_[idx].next_phys = nn;
      if (nn != kNullIdx) nodes_[nn].prev_phys = idx;
      else tail_idx_ = idx;
      RecycleSpanNode(next);
    }
    InsertFreeSpan(idx);
  }

  std::optional<Allocation> AllocateBuddy(uint64_t size, uint64_t alignment) {
    uint64_t pow2_size = 0;
    if (!NextPowerOfTwo(size, &pow2_size)) return std::nullopt;
    const uint64_t base_bytes = std::max(min_block_bytes_, pow2_size);
    const uint64_t req_bytes = std::max(base_bytes, alignment);
    if (req_bytes > capacity_bytes_) return std::nullopt;
    const size_t target_order = FloorLog2(req_bytes / min_block_bytes_);
    size_t order = target_order;
    while (order <= max_order_ && ((buddy_free_mask_ >> order) & 1ULL) == 0) ++order;
    if (order > max_order_) return std::nullopt;

    const uint64_t offset = *buddy_free_[order].begin();
    buddy_free_[order].erase(buddy_free_[order].begin());
    if (buddy_free_[order].empty()) buddy_free_mask_ &= ~(uint64_t{1} << order);
    while (order > target_order) {
      --order;
      buddy_free_[order].insert(offset + (min_block_bytes_ << order));
      buddy_free_mask_ |= (uint64_t{1} << order);
    }
    Allocation alloc{next_handle_++, offset, offset, size, req_bytes,
                     alignment, req_bytes - base_bytes};
    buddy_live_.emplace(alloc.handle, alloc);
    return alloc;
  }

  void DeallocateBuddyBlock(uint64_t offset, uint64_t block_bytes) {
    size_t order = FloorLog2(block_bytes / min_block_bytes_);
    while (order < max_order_) {
      const uint64_t buddy_off = offset ^ (min_block_bytes_ << order);
      auto it = buddy_free_[order].find(buddy_off);
      if (it == buddy_free_[order].end()) break;
      buddy_free_[order].erase(it);
      if (buddy_free_[order].empty()) buddy_free_mask_ &= ~(uint64_t{1} << order);
      offset = std::min(offset, buddy_off);
      ++order;
    }
    buddy_free_[order].insert(offset);
    buddy_free_mask_ |= (uint64_t{1} << order);
  }

  uint64_t LargestFreeBlock() const noexcept {
    if (strategy_ == Strategy::kBuddy) {
      return (buddy_free_mask_ == 0) ? 0 : (min_block_bytes_ << FloorLog2(buddy_free_mask_));
    }
    if (non_empty_bins_ == 0) return 0;
    const size_t top = FloorLog2(non_empty_bins_);
    if (strategy_ == Strategy::kFreeListBestFit) return bins_[top].rbegin()->primary;
    uint64_t best = 0;
    for (const BinEntry& e : bins_[top]) best = std::max(best, e.secondary);
    return best;
  }

  void RefreshFragmentation() noexcept {
    const uint64_t free_bytes = capacity_bytes_ - live_reserved_bytes_;
    current_external_fragmentation_ =
        (free_bytes == 0)
            ? 0.0
            : (1.0 - static_cast<double>(LargestFreeBlock()) / static_cast<double>(free_bytes));
    peak_external_fragmentation_ =
        std::max(peak_external_fragmentation_, current_external_fragmentation_);
  }

  bool VerifyFreeListInvariants(std::string* err) const {
    auto fail = [&](const std::string& m) { if (err) *err = m; return false; };
    if (head_idx_ == kNullIdx || tail_idx_ == kNullIdx) return fail("Empty span chain");
    uint64_t expected_off = 0, sum_free = 0, sum_res = 0;
    size_t free_cnt = 0, live_cnt = 0;
    uint32_t prev = kNullIdx, cur = head_idx_;
    bool prev_free = false;
    while (cur != kNullIdx) {
      if (cur >= nodes_.size() || !nodes_[cur].in_use) return fail("Inactive span in list");
      const SpanNode& n = nodes_[cur];
      if (n.prev_phys != prev || n.offset != expected_off || n.length == 0) {
        return fail("Span contiguity invariant broken");
      }
      expected_off += n.length;
      if (n.is_free) {
        if (prev_free) return fail("Uncoalesced adjacent free spans");
        if (bins_[FloorLog2(n.length)].count(MakeBinEntry(cur)) == 0) return fail("Missing bin");
        sum_free += n.length; ++free_cnt;
      } else {
        const Allocation& a = n.alloc;
        if (!IsPowerOfTwo(a.alignment) || (a.payload_offset % a.alignment) != 0 ||
            a.raw_offset != n.offset || a.reserved_size != n.length ||
            a.payload_offset != a.raw_offset + a.alignment_padding ||
            a.reserved_size != a.alignment_padding + a.payload_size) {
          return fail("Allocated span alignment or size mismatch");
        }
        auto it = freelist_live_.find(a.handle);
        if (it == freelist_live_.end() || it->second != cur) return fail("Handle mismatch");
        sum_res += n.length; ++live_cnt;
      }
      prev_free = n.is_free; prev = cur; cur = n.next_phys;
    }
    if (prev != tail_idx_ || expected_off != capacity_bytes_ ||
        live_cnt != freelist_live_.size() || sum_res != live_reserved_bytes_ ||
        sum_free + sum_res != capacity_bytes_) return fail("Partition accounting mismatch");
    size_t binned = 0;
    uint64_t mask = 0;
    for (size_t b = 0; b < kNumBins; ++b) {
      if (!bins_[b].empty()) mask |= (uint64_t{1} << b);
      binned += bins_[b].size();
    }
    return (mask == non_empty_bins_ && binned == free_cnt) ? true : fail("Bin mask mismatch");
  }

  bool VerifyBuddyInvariants(std::string* err) const {
    auto fail = [&](const std::string& m) { if (err) *err = m; return false; };
    std::vector<std::pair<uint64_t, uint64_t>> segs;
    uint64_t sum_res = 0, sum_free = 0, mask = 0;
    for (const auto& [h, a] : buddy_live_) {
      (void)h;
      if (!IsPowerOfTwo(a.reserved_size) || a.reserved_size < min_block_bytes_ ||
          (a.raw_offset % a.reserved_size) != 0 || (a.payload_offset % a.alignment) != 0) {
        return fail("Invalid live buddy geometry");
      }
      segs.emplace_back(a.raw_offset, a.reserved_size);
      sum_res += a.reserved_size;
    }
    for (size_t order = 0; order <= max_order_; ++order) {
      if (!buddy_free_[order].empty()) mask |= (uint64_t{1} << order);
      const uint64_t bsz = min_block_bytes_ << order;
      for (uint64_t off : buddy_free_[order]) {
        if ((off % bsz) != 0 || off + bsz > capacity_bytes_) return fail("Bad free buddy");
        if (order < max_order_ && buddy_free_[order].count(off ^ bsz) != 0) {
          return fail("Uncoalesced free buddy pair");
        }
        segs.emplace_back(off, bsz);
        sum_free += bsz;
      }
    }
    if (mask != buddy_free_mask_) return fail("Buddy bitmask mismatch");
    std::sort(segs.begin(), segs.end());
    uint64_t cur = 0;
    for (const auto& [off, len] : segs) {
      if (off != cur) return fail("Buddy partition gap or overlap");
      cur += len;
    }
    return (cur == capacity_bytes_ && sum_res == live_reserved_bytes_ &&
            sum_res + sum_free == capacity_bytes_) ? true : fail("Buddy coverage mismatch");
  }

  uint64_t capacity_bytes_ = 0;
  Strategy strategy_ = Strategy::kFreeListFirstFit;
  uint64_t min_block_bytes_ = 16;
  size_t max_order_ = 0;
  uint64_t next_handle_ = 1;
  std::vector<SpanNode> nodes_;
  std::vector<uint32_t> recycled_nodes_;
  uint32_t head_idx_ = kNullIdx, tail_idx_ = kNullIdx;
  std::array<std::set<BinEntry>, kNumBins> bins_;
  uint64_t non_empty_bins_ = 0;
  std::unordered_map<uint64_t, uint32_t> freelist_live_;
  std::vector<std::set<uint64_t>> buddy_free_;
  uint64_t buddy_free_mask_ = 0;
  std::unordered_map<uint64_t, Allocation> buddy_live_;
  std::unordered_set<uint64_t> retired_handles_;
  uint64_t accepted_ = 0, failed_ = 0, freed_ = 0, invalid_frees_ = 0, double_frees_ = 0;
  uint64_t live_reserved_bytes_ = 0, live_payload_bytes_ = 0;
  uint64_t peak_reserved_bytes_ = 0, peak_payload_bytes_ = 0;
  uint64_t alignment_overhead_bytes_ = 0, live_alignment_overhead_bytes_ = 0;
  uint64_t peak_alignment_overhead_bytes_ = 0;
  double current_external_fragmentation_ = 0.0, peak_external_fragmentation_ = 0.0;
};

}  // namespace slab_arena

#endif  // WAVE_SLAB_ARENA_HPP_
