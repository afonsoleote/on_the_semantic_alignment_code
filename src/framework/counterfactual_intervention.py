from .symbolic_cf import SymbolicCF

"""I = <S, R>"""
def counterfactual_intervention(x, V, a, K, R, S=SymbolicCF(), seed=0):
    V_cf = S(V, a, K, seed)
    x_cf = R(x, V, V_cf)
    return x_cf

def labeling(df, x, attributes):
    """V(x): the attribute values of x in D."""
    return {a: bool(df.loc[x, a]) for a in attributes}