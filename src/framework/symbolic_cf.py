import itertools
import random


class SymbolicCF:
    def flip(self, V, attributes):
        W = dict(V)
        for a in attributes:
            W[a] = not W[a]
        return W

    def distance(self, V, W):
        return frozenset(a for a in V if V[a] != W[a])

    def minimal(self, V, candidates):
        ds = [self.distance(V, W) for W in candidates]
        return [W for W, d in zip(candidates, ds) if not any(e < d for e in ds)]

    def repairs(self, V, flipped, K):
        found = []
        frontier = {frozenset(flipped)}
        while frontier:
            next_frontier = set()
            for F in frontier:
                if any(self.distance(V, W) <= F for W in found):
                    continue
                W = self.flip(V, F)
                broken = [f for f in K if not f.eval(W)]
                if not broken:
                    found.append(W)
                else:
                    for b in broken[0].atoms() - F:
                        next_frontier.add(F | {b})
            frontier = next_frontier
        return found

    def candidates(self, V, a, K, extra=0):
        out = {}
        others = sorted(b for b in V if b != a)
        for chosen in itertools.combinations(others, extra):
            for W in self.minimal(V, self.repairs(V, [a, *chosen], K)):
                out[self.distance(V, W)] = W
        return [out[d] for d in sorted(out, key=lambda d: (len(d), sorted(d)))]

    def select(self, candidates, seed):
        return random.Random(seed).choice(candidates)

    def __call__(self, V, a, K, seed=0):
        return self.select(self.candidates(V, a, K), seed)