from collections import defaultdict
from dataclasses import dataclass
import math
from enum import Enum, auto
from typing import Any

class Ops(Enum):
    REG = auto()
    MOV = auto()
    PUSH = auto()
    POP = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    LOAD = auto()
    STORE = auto()

class UOp:
    def __init__(self, op: Ops, srcs: list[Any] = []):
        self.op = op
        self.value = None
        self.srcs = srcs or []

    def __repr__(self):
        if self.srcs:
            return f"UOp({self.op.name}, srcs={self.srcs})"
        return f"UOp({self.op.name})"

@dataclass(frozen=True)
class Register:
    name: str
    size: int

def log2i(x: int) -> int:
    assert x > 0 and (x & (x - 1)) == 0, f"{x} is not a power of 2"
    return int(math.log2(x))

@dataclass(frozen=True)
class Memory:
    base: str | None = None
    index: str | None = None
    scale: int = 1
    offset: int = 0
    size: int = 1

    def __str__(self) -> str:
        parts = []

        if self.base:
            parts.append(self.base)

        if self.index:
            if self.scale != 1:
                parts.append(f"{self.index}*{self.scale}")
            else:
                parts.append(self.index)

        if self.offset:
            sign = "+" if self.offset > 0 else "-"
            parts.append(f"{sign}{hex(abs(self.offset))}")

        if not parts:
            return f"[{hex(self.offset)}]"

        access = ("BYTE", "WORD", "DWORD", "QWORD")[log2i(self.size)]
        return f"{access} [{' + '.join(parts)}]"

@dataclass(frozen=True)
class MemoryChunk:
    offset: int
    size: int
    value: UOp | int | None

class IntervalMap:
    def __init__(self):
        self.intervals = []

    def set(self, start: int, size: int, value):
        end = start + size
        new_intervals = []
        for s, e, v in self.intervals:
            if e <= start or s >= end:  # No overlap
                new_intervals.append((s, e, v))
            else:  # Overlap - split if needed
                if s < start:  # Keep left part
                    new_intervals.append((s, start, v))
                if e > end:    # Keep right part
                    new_intervals.append((end, e, v))

        if value is not None:
            new_intervals.append((start, end, value))
        self.intervals = sorted(new_intervals)

    def get(self, start: int, size: int):
        end = start + size
        overlapping = [(s, e, v) for s, e, v in self.intervals
                      if not (e <= start or s >= end)]

        if not overlapping:
            return None

        overlapping.sort()

        coverage_start = min(s for s, e, v in overlapping)
        coverage_end = max(e for s, e, v in overlapping)

        if coverage_start <= start and coverage_end >= end:
            first_val = overlapping[0][2]
            if first_val is not None and all(v == first_val for _, _, v in overlapping):
                overlapping.sort()
                for i in range(len(overlapping) - 1):
                    if overlapping[i][1] < overlapping[i + 1][0]:
                        return None
                return (first_val >> ((start - coverage_start) * 8)) % (1 << (8 * size))

        return None

    def clear(self):
        self.intervals.clear()

class SparseEnv:
    def __init__(self):
        self._regs: dict[str, UOp | int | None] = {}
        self._mem: dict[str | None, IntervalMap] = defaultdict(IntervalMap)

    def _effective_offset(self, mem: Memory) -> int | None:
        offset = mem.offset
        if mem.index:
            if isinstance(mem.index, str):  # Symbolic
                return None
            offset += mem.index * mem.scale
        return offset

    def __getitem__(self, key: Register| Memory) -> UOp| int | None:
        if isinstance(key, Register):
            return self._regs.get(key.name)

        offset = self._effective_offset(key)
        if offset is None:
            return None

        return self._mem[key.base].get(offset, key.size)

    def __setitem__(self, key: Register | Memory, value: UOp | int | None):
        if isinstance(key, Register):
            self._regs[key.name] = value
            return

        offset = self._effective_offset(key)
        if offset is None:
            self._mem[key.base].clear()
            return

        self._mem[key.base].set(offset, key.size, value)

Symbol = Memory | Register
