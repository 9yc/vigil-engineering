"""Capacity bookkeeping for fixed slots, multiple blocks and contiguous extents.

Units are abstract integers. This module accounts for occupied ranges; it does
not allocate payload memory, move data or choose which objects to discard.
"""

from dataclasses import dataclass


def _positive_integer(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True, eq=False)
class Allocation:
    requested_units: int
    reserved_units: int
    segments: tuple


class CapacityPool:
    """Allocate on demand without compaction, eviction or future knowledge."""

    def __init__(self, budget_units, *, mode="contiguous", block_size=1):
        self.budget_units = _positive_integer(budget_units, "budget_units")
        self.block_size = _positive_integer(block_size, "block_size")
        if mode not in ("fixed", "blocks", "contiguous"):
            raise ValueError("mode must be fixed, blocks or contiguous")
        if mode == "contiguous" and block_size != 1:
            raise ValueError("contiguous mode uses unit granularity")
        self.mode = mode
        if mode == "contiguous":
            self.capacity = self.budget_units
            self._free_extents = [(0, self.capacity)]
        else:
            count = self.budget_units // self.block_size
            self.capacity = count * self.block_size
            self._free_blocks = list(range(count))
        self._live = {}
        self._payload = self._held = 0
        self._peak_payload = self._peak_held = 0
        self._requests = self._accepted = 0
        self._failures = dict(capacity=0, rounding=0, fragmentation=0, object_limit=0)

    def _fail(self, reason):
        self._failures[reason] += 1
        return None

    def allocate(self, units):
        """Return a pool-owned Allocation, or None without partially allocating."""
        units = _positive_integer(units, "units")
        self._requests += 1
        if self.mode != "contiguous":
            if self.mode == "fixed" and units > self.block_size:
                return self._fail("object_limit")
            count = 1 if self.mode == "fixed" else (units + self.block_size - 1) // self.block_size
            if count > len(self._free_blocks):
                reason = "rounding" if units <= self.capacity - self._payload else "capacity"
                return self._fail(reason)
            self._free_blocks.sort()
            blocks = self._free_blocks[:count]
            del self._free_blocks[:count]
            segments = tuple((index * self.block_size, self.block_size) for index in blocks)
            held = count * self.block_size
        else:
            for index, (offset, length) in enumerate(self._free_extents):
                if length >= units:
                    segments = ((offset, units),)
                    if length == units:
                        del self._free_extents[index]
                    else:
                        self._free_extents[index] = (offset + units, length - units)
                    break
            else:
                reason = "fragmentation" if units <= self.capacity - self._held else "capacity"
                return self._fail(reason)
            held = units
        allocation = Allocation(units, held, segments)
        self._live[allocation] = allocation
        self._payload += units
        self._held += held
        self._accepted += 1
        self._peak_payload = max(self._peak_payload, self._payload)
        self._peak_held = max(self._peak_held, self._held)
        return allocation

    def free(self, allocation):
        """Return one live allocation; reject foreign handles and repeated frees."""
        try:
            record = self._live.pop(allocation)
        except (KeyError, TypeError) as exc:
            raise ValueError("not a live allocation from this pool") from exc
        if self.mode != "contiguous":
            self._free_blocks.extend(offset // self.block_size
                                     for offset, _ in record.segments)
        else:
            extents = sorted(self._free_extents + list(record.segments))
            merged = []
            for offset, length in extents:
                if merged and merged[-1][0] + merged[-1][1] == offset:
                    merged[-1] = (merged[-1][0], merged[-1][1] + length)
                else:
                    merged.append((offset, length))
            self._free_extents = merged
        self._payload -= record.requested_units
        self._held -= record.reserved_units

    def stats(self):
        return {
            "mode": self.mode,
            "budget_units": self.budget_units,
            "usable_units": self.capacity,
            "unallocatable_tail_units": self.budget_units - self.capacity,
            "requests": self._requests,
            "accepted": self._accepted,
            "failed": sum(self._failures.values()),
            "failures_by_reason": dict(self._failures),
            "live_allocations": len(self._live),
            "payload_units": self._payload,
            "reserved_units": self._held,
            "free_units": self.capacity - self._held,
            "internal_waste_units": self._held - self._payload,
            "peak_payload_units": self._peak_payload,
            "peak_reserved_units": self._peak_held,
        }

    def check_invariants(self):
        """Check accounting and an exact non-overlapping partition of capacity."""
        live = list(self._live.values())
        assert sum(item.requested_units for item in live) == self._payload
        assert sum(item.reserved_units for item in live) == self._held
        assert 0 <= self._payload <= self._held <= self.capacity <= self.budget_units
        assert self._payload <= self._peak_payload <= self._peak_held <= self.capacity
        assert self._held <= self._peak_held
        assert self._requests == self._accepted + sum(self._failures.values())
        allocated = [extent for item in live for extent in item.segments]
        for item in live:
            assert item.requested_units > 0
            assert sum(length for _, length in item.segments) == item.reserved_units
        if self.mode == "contiguous":
            free = list(self._free_extents)
            assert all(a + length < b for (a, length), (b, _) in zip(free, free[1:]))
        else:
            free = [(index * self.block_size, self.block_size)
                    for index in self._free_blocks]
            assert all(offset % self.block_size == 0 and length == self.block_size
                       for offset, length in allocated + free)
        cursor = 0
        for offset, length in sorted(allocated + free):
            assert offset == cursor and length > 0
            cursor = offset + length
        assert cursor == self.capacity
        assert sum(length for _, length in free) == self.capacity - self._held
