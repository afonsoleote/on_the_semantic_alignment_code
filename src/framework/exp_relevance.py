"""Which attributes enter the query of each probe.

S only flips attributes that occur in a formula of K made false, so flipping
an attribute can only change attributes linked to it by a chain of formulas.
An attribute is irrelevant to a concept C if no chain of formulas of
K_C = K ∪ {C ↔ φ_C} links them.
"""
import itertools

from src.framework.symbolic_cf import SymbolicCF
from src.utils.propositional_logic import Atom


def linked(attribute, K):
    """All attributes linked to `attribute` by a chain of formulas of K."""
    found = [attribute]
    todo = [attribute]
    while todo:
        b = todo.pop()
        for f in K:
            if b in f.atoms():
                for c in f.atoms():
                    if c not in found:
                        found.append(c)
                        todo.append(c)
    return found


def irrelevant(concept, setup):
    """Attributes of A not linked to the concept through K_C."""
    linked_to_concept = linked(concept, setup.knowledge(concept))
    result = []
    for a in setup.attributes:
        if a not in linked_to_concept:
            result.append(a)
    return result


def query_attributes(concept, setup):
    """Dependence: the concept itself (its auxiliary attribute a_C).
    Independence: the irrelevant attributes.
    Neither: attributes linked to C through K but not part of its definition."""
    independence = irrelevant(concept, setup)
    defining = sorted(setup.concepts[concept].atoms())
    neither = []
    for a in setup.attributes:
        if a not in independence and a not in defining:
            neither.append(a)
    return defining, independence, neither


def labelings_where(formula, setup):
    """All labelings of A that are consistent with K and satisfy the formula."""
    result = []
    for values in itertools.product([False, True], repeat=len(setup.attributes)):
        V = dict(zip(setup.attributes, values))
        if formula.eval(V) and all(f.eval(V) for f in setup.K):
            result.append(V)
    return result


def query_attributes_positive(concept, setup, given='concept', exclude=(), entangled=(),
                              S=SymbolicCF()):
    """Which attributes enter the query of the probe for the concept.
    `given` says which inputs the atoms are evaluated on:
      'concept':   inputs where the concept holds (C = 1)
      'attribute': inputs where the intervened attribute holds (a = 1), whatever C is;
                   for realizations that can only remove attributes (GTSRB, CUB)
      dependence:   the concept itself, through its auxiliary attribute a_C defined by
                    C ↔ φ_C in K_C, and every attribute of φ_C whose flip changes C
                    in every such labeling (e.g. each conjunct of an AND concept)
      independence: attributes whose flip never changes C, in every such labeling and
                    every minimal counterfactual S gives for them
      left out:     all other attributes
    Two lists cover what K cannot express, because it comes from how the input is
    edited rather than from logic:
      exclude:   attributes that cannot be intervened on without changing other
                 content of the input (e.g. removing a wagon)
      entangled: groups of attributes that cannot be edited one without the other
                 (e.g. AttGAN changes the gender when it adds a beard); an attribute
                 in a group with C or with an attribute of φ_C is left out
    The counterfactuals use K alone: when intervening on an attribute, the concept
    only follows its definition, it does not constrain the other attributes."""
    phi = setup.concepts[concept]
    defining = set(phi.atoms())
    defining.add(concept)

    # dependence on the concept itself
    dependence = [concept]

    # dependence on the attributes of the definition that always change the concept;
    # comment out this block to keep only the dependence on the concept itself
    for a in sorted(phi.atoms()):
        if a == concept:
            continue
        always = True
        for V in inputs(a, phi, given, setup):
            for W in S.candidates(V, a, setup.K):
                if phi.eval(W) == phi.eval(V):
                    always = False
        if always:
            dependence.append(a)

    independence, left_out = [], []
    for a in setup.attributes:
        if a == concept or a in dependence:
            continue
        if a in exclude:
            left_out.append(a)
            continue
        tied = False
        for group in entangled:
            if a in group and len((group - {a}) & defining) > 0:
                tied = True
        if tied:
            left_out.append(a)
            continue
        never = True
        for V in inputs(a, phi, given, setup):
            for W in S.candidates(V, a, setup.K):
                if phi.eval(W) != phi.eval(V):
                    never = False
        if never:
            independence.append(a)
        else:
            left_out.append(a)
    return dependence, independence, left_out


def inputs(a, phi, given, setup):
    """The labelings on which an intervention on a is evaluated."""
    if given == 'attribute':
        return labelings_where(Atom(a), setup)
    return labelings_where(phi, setup)


if __name__ == '__main__':
    from configs.experimental_setup import (AdultSetup, CelebASetup, CUBSetup, GTSRBSetup,
                                            XTRAINSSetup)

    SETUPS = {
        'XTRAINS': XTRAINSSetup,
        'GTSRB': GTSRBSetup,
        'CUB': CUBSetup,
        'CelebA': CelebASetup,
        'Adult': AdultSetup,
    }

    # removing or adding a wagon changes the content of the train
    EXCLUDE = {
        'XTRAINS': ['three_wagons'],
    }

    # which inputs the atoms are evaluated on: realizations that can only remove
    # attributes are evaluated where the intervened attribute holds
    GIVEN = {
        'GTSRB': 'attribute',
    }

    # attributes that the realization cannot edit one without the other
    ENTANGLED = {
        'GTSRB': [
            {'Symbol', 'Symbol_Speed80', 'Black_Symbol'},  # a sign has one symbol: removing
                                                           # the 80 or the black symbol removes it
            {'Black_Bar', 'Symbol', 'Black_Symbol'},       # the bar is drawn over the symbol
        ],
        'CelebA': [
            {'Male', 'Mustache', 'No_Beard'},   # AttGAN changes the gender with facial hair
            {'Young', 'Mustache', 'No_Beard'},  # and the apparent age
        ],
    }

    for name, setup in SETUPS.items():
        print(f'===== {name} =====')
        for concept in setup.concepts:
            given = GIVEN.get(name, 'concept')
            dependence, independence, left_out = query_attributes_positive(
                concept, setup, given, EXCLUDE.get(name, []), ENTANGLED.get(name, []))
            if given == 'attribute':
                print(f'{concept}  (assuming the intervened attribute = 1)')
            else:
                print(f'{concept}  (assuming {concept} = 1)')
            print(f'  dependence:   {", ".join(dependence)}   ({concept} ↔ {", ".join(sorted(setup.concepts[concept].atoms()))})')
            print(f'  independence: {", ".join(independence) if independence else "-"}')
            print(f'  left out:     {", ".join(left_out) if left_out else "-"}')
        print()