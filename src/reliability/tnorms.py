from functools import reduce
from math import prod



# Basic t-norms

def tnorm_min(values):
    return min(values)


def tnorm_product(values):
    return prod(values)


# Hamacher t-norm

def hamacher_pair(x, y):
    if x == 0 and y == 0:
        return 0.0
    return (x * y) / (x + y - x * y)


def tnorm_hamacher(values):
    return reduce(hamacher_pair, values)


# Yager t-norm

def yager_pair(x, y, p=2):
    return 1 - min(1.0, ((1 - x) ** p + (1 - y) ** p) ** (1 / p))


def tnorm_yager(values, p=2):
    return reduce(lambda a, b: yager_pair(a, b, p), values)

if __name__ == "__main__":
    values = [0.95, 0.92, 0.91, 0.3]

    print("values:", values)

    print("min      :", tnorm_min(values))
    print("product  :", tnorm_product(values))
    print("hamacher :", tnorm_hamacher(values))
    print("yager p=2:", tnorm_yager(values, p=2))
    print("yager p=5:", tnorm_yager(values, p=5))
    print("yager p=10:", tnorm_yager(values, p=10))