class Realization:
    attributes = []

    def changes(self, V, Vcf):
        return [(a, Vcf[a]) for a in self.attributes if V[a] != Vcf[a]]

    def realize(self, x, V, Vcf):
        raise NotImplementedError

    def __call__(self, x, V, Vcf):
        return self.realize(x, V, Vcf)