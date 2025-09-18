from ops import Ops, Register, UOp, Symbol, SparseEnv, Memory

from typing import Callable, Any
from collections import defaultdict, deque

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
        overlap = (g1.wo | g1.rw) & (g2.ro | g2.rw)
        if overlap:
            graph[g1].append((g2, overlap))

for g, succs in graph.items():
    print(f"{g.code} -> {[s[0].code for s in succs]}")

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
            elif e == "*":
                continue
            else:
                if e != src:
                    return False

        return True

class PatternMatcher:
    def __init__(self, pats: dict[Pattern, Callable]):
        self._pats = defaultdict(list)
        for pat, f in pats.items():
            for op in pat.ops:
                self._pats[(op, len(pat.srcs or []))].append((pat, f))

    def rewrite_all(self, uops: list[UOp]) -> list[UOp]:
        out = []
        subst: dict[UOp, UOp] = {}

        for uop in uops:
            new_srcs = [subst.get(s, s) for s in uop.srcs]

            replaced = False
            for pat, f in self._pats[(uop.op, len(uop.srcs))]:
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

    def __call__(self, uop: UOp, subst: dict[UOp, UOp]) -> Any:
        new_srcs = [subst.get(s, s) for s in uop.srcs]
        for pat, f in self._pats[(uop.op, len(uop.srcs))]:
            if pat.matches(uop):
                r = f(uop.op, *new_srcs)
                if r is None:
                    continue

                subst[uop] = r
                return r

rax = Register("rax", size=8)

u1 = UOp(Ops.SUB, [rax, 1])
u2 = UOp(Ops.SUB, [rcx, 5])
u3 = UOp(Ops.SUB, [u1, u2])

uops = [u1, u2, u3]

def handle_sub(op, lhs, rhs):
    if isinstance(lhs, int) and lhs == 0:
        # 0 - x => -x
        return UOp(Ops.SUB, [0, rhs])

    if isinstance(rhs, int) and rhs == 0:
        # x - 0 => x
        return lhs

    return UOp(Ops.SUB, [lhs, rhs])

# last uop is assumed to be the output
def gc(uops: list[UOp]) -> list[UOp]:
    SIDE_EFFECT_OPS = {Ops.STORE, Ops.POP, Ops.PUSH}

    live = set()
    out = []

    stack = [uops[-1]]
    for uop in uops:
        if not isinstance(uop, UOp): continue

        if uop in live or uop.op in SIDE_EFFECT_OPS:
            stack.append(uop)

    live.add(uops[-1])
    while stack:
        uop = stack.pop()
        if not isinstance(uop, UOp): continue
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
    Pattern([Ops.SUB], srcs=[int, int]): lambda o,lhs,rhs: lhs - rhs,
    Pattern([Ops.SUB], srcs=[UOp, UOp]): lambda o,lhs,rhs: 0 if lhs == rhs else None,
    Pattern([Ops.SUB], srcs=[Register, Register]): lambda o,lhs,rhs: 0 if lhs == rhs else None,
    Pattern([Ops.ADD], srcs=[int, int]): lambda o,lhs,rhs: lhs + rhs,
})

import time
start = time.perf_counter()
r = gc(matcher.rewrite_all(uops))
end = time.perf_counter()
print(f"Pattern matching took {(end-start)*1e6:.1f} us")

print(r)

# index gadgets by their operations. have priorities + temperature

def symeval(uops: list[UOp], env: SparseEnv, matcher: PatternMatcher) -> list[UOp]:
    out = []
    subst = {}

    for u in uops:
        new_srcs = []
        for s in u.srcs:
            if isinstance(s, Memory) or isinstance(s, Register):
                v = env[s]
                new_srcs.append(v if v is not None else s)
            elif s in subst and subst[s] is not None:
                new_srcs.append(subst[s])
            else:
                new_srcs.append(s)

        u_new = UOp(u.op, new_srcs)
        u_simplified = matcher(u_new, {})
        out.append(u_simplified if u_simplified is not None else u_new)

        subst[u] = u_simplified
        if isinstance(u.srcs[0], Symbol):
            env[u.srcs[0]] = u_simplified

    return out

# movement gadgets
# compute gadgets

env = SparseEnv()
env[rax] = 4
env[rcx] = 5
print(uops)
symeval(uops, env, matcher)
print(env)
# we want to score stuff based on its compute and movement. basically what it captures, its SparseEnv after it completes


