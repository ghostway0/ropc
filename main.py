from ops import Ops, Register, UOp, Symbol, SparseEnv

from typing import Callable, Any
from collections import defaultdict

# connect all rop gadgets that use the same things.

# I need aliasing analysis: kill the stack at *sp+20 or *rbp+30, they may turn to be the same thing.
# yeah so I basically need to try and alias based on the env on every step?

# I also need independence analysis to see what I can do independently. but that's later

# when I try to construct an operation (a list[UOp]) I must also track the state. this is where the aliasing analysis comes to be with the symbols map

class Gadget:
    def __init__(self, ro: set[Symbol], rw: set[Symbol], wo: set[Symbol], code: list[UOp]) -> None:
        self.ro = ro
        self.rw = rw
        self.wo = wo
        self.code = code


rcx = Register("rcx", size=8)
rax = Register("rax", size=8)
rsp = Register("rsp", size=8)

gadgets = [
    Gadget(
        ro=set(),
        rw={rcx},
        wo=set(),
        code=[UOp(Ops.SUB, [rcx, 12])]
    ),

    Gadget(
        ro=set(),
        rw={rcx},
        wo=set(),
        code=[UOp(Ops.ADD, [rcx, 1])]
    ),

    Gadget(
        ro={rcx},
        rw=set(),
        wo={rax},
        code=[UOp(Ops.MOV, [rax, rcx])]
    ),
]

graph: dict[Gadget, list[Gadget]] = defaultdict(list)

for g1 in gadgets:
    for g2 in gadgets:
        if g1 is g2:
            continue
        if g1.wo & (g2.ro | g2.wo | g2.rw):
            graph[g1].append(g2)

for g, succs in graph.items():
    print(f"{g.code} -> {[s.code for s in succs]}")

a0 = Register("a0", size=8)
entry, code = {"a0": rax}, [UOp(Ops.SUB, [a0, 1])]

# two things to think about:
# dataflow - the movement of data between ops
# op similarity - operation similarity between what we have and the wanted op
#
# we may have hundreds of thousands of rop gadgets. we can't just evaluate complexities at every step.
# op type -> gadgets then we & with the graph available dataflow gadgets
# then from that we have to use rewrites to actually resolve our ops to evaluate it


class Pattern:
    def __init__(self, ops: list[Ops], srcs: list[Any] | None = None):
        self.ops = ops
        self.srcs = srcs

    def matches(self, uop: UOp) -> bool:
        if not self.ops or uop.op not in self.ops:
            return False

        if self.srcs is not None and len(self.srcs) != len(uop.srcs):
            return False

        if self.srcs is None:
            return True

        for e, src in zip(self.srcs, uop.srcs):
            # if it's of type Ops, check if we have the same op
            # if it's of type UOp we do this recursively
            # if it's a type, check if it's an instance of the type or the type itself
            # if it's a constant, check if we are equal to it
            # etc.
            if isinstance(e, Ops):
                if isinstance(src, Ops) and src != e:
                    return False
                elif isinstance(src, UOp):
                    if src.op != e:
                        return False
                else:
                    return False
            elif isinstance(e, UOp):
                if not isinstance(src, UOp):
                    return False
                if not Pattern([e.op], e.srcs).matches(src):
                    return False
            elif isinstance(e, type):
                if src != e:
                    if not isinstance(src, e):
                        return False
            else:
                if e != src:
                    return False

        return True

class PatternMatcher:
    def __init__(self, pats: dict[Pattern, Callable]):
        self._pats = pats

    def __call__(self, uops: list[UOp]) -> list[UOp]:
        out = []
        subst: dict[UOp, UOp] = {}

        for uop in uops:
            new_srcs = [subst.get(s, s) for s in uop.srcs]

            replaced = False
            for pat, f in self._pats.items():
                if pat.matches(uop):
                    r = f(uop.op, *new_srcs)
                    if r is None:
                        continue

                    subst[uop] = r
                    out.append(r)
                    replaced = True
                    break

            if not replaced:
                out.append(UOp(uop.op, new_srcs))

        return out

rax = Register("rax", size=8)

u1 = UOp(Ops.SUB, [rax, 1])
u2 = UOp(Ops.SUB, [rax, 2])
u3 = UOp(Ops.SUB, [u1, u2])

uops = [u1, u2, u3]

def handle_sub(op, lhs, rhs):
    new_srcs = []

    const_sum = 0

    def collect(u):
        nonlocal const_sum
        if isinstance(u, int):
            const_sum += u
        elif isinstance(u, UOp) and u.op == Ops.SUB:
            for s in u.srcs:
                collect(s)
        else:
            new_srcs.append(u)

    collect(lhs)
    collect(rhs)

    if const_sum != 0:
        new_srcs.append(const_sum)

    return UOp(Ops.SUB, new_srcs)

# last uop is assumed to be the output
def gc(uops: list[UOp]) -> list[UOp]:
    SIDE_EFFECT_OPS = {Ops.STORE, Ops.POP, Ops.PUSH}

    live = set()
    out = []

    stack = [uops[-1]]
    for uop in uops:
        if uop in live or uop.op in SIDE_EFFECT_OPS:
            stack.append(uop)

    live.add(uops[-1])
    while stack:
        uop = stack.pop()
        for src in uop.srcs:
            if isinstance(src, UOp) and src not in live:
                live.add(src)
                stack.append(src)

    for uop in uops:
        if uop in live:
            out.append(uop)

    return out

matcher = PatternMatcher({
    Pattern([Ops.SUB], srcs=[Ops.SUB, Ops.SUB]): handle_sub,
})

import time
start = time.perf_counter()
r = gc(matcher(uops))
end = time.perf_counter()
print(f"Pattern matching took {(end-start)*1e6:.1f} us")

print(r)

# index gadgets by their operations. have priorities + temperature
