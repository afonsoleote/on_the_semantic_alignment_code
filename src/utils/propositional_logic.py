class Atom:
    def __init__(self, name):
        self.name = name

    def eval(self, V):
        return V[self.name]

    def atoms(self):
        return {self.name}


class Not:
    def __init__(self, f):
        self.f = f

    def eval(self, V):
        return not self.f.eval(V)

    def atoms(self):
        return self.f.atoms()


class And:
    def __init__(self, *fs):
        self.fs = fs

    def eval(self, V):
        return all(f.eval(V) for f in self.fs)

    def atoms(self):
        return set().union(*(f.atoms() for f in self.fs))


class Or:
    def __init__(self, *fs):
        self.fs = fs

    def eval(self, V):
        return any(f.eval(V) for f in self.fs)

    def atoms(self):
        return set().union(*(f.atoms() for f in self.fs))


class Implies:
    def __init__(self, f, g):
        self.f, self.g = f, g

    def eval(self, V):
        return not self.f.eval(V) or self.g.eval(V)

    def atoms(self):
        return set().union(self.f.atoms(), self.g.atoms())


class Iff:
    def __init__(self, f, g):
        self.f, self.g = f, g

    def eval(self, V):
        return self.f.eval(V) == self.g.eval(V)

    def atoms(self):
        return set().union(self.f.atoms(), self.g.atoms())



if __name__ == "__main__":

    P = Atom("P")
    Q = Atom("Q")
    R = Atom("R")

    V = {
        "P": True,
        "Q": False,
        "R": True
    }

    print("Atom:", P.eval(V))
    print("Not:", Not(P).eval(V))
    print("And:", And(P, Q).eval(V))
    print("Or:", Or(P, Q).eval(V))
    print("Implies:", Implies(P, Q).eval(V))
    print("Iff:", Iff(P, Q).eval(V))

    print("\n")
    print("Atoms in And:", And(P, Q).atoms())
    print("Atoms in Implies:", Implies(P, Q).atoms())

    formula = Implies(And(P, Q), Not(R))

    print("\n")
    print("Complex formula:")
    print("Atoms:", formula.atoms())
    print("Evaluation:", formula.eval(V))

