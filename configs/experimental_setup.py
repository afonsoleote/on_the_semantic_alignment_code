from dataclasses import dataclass

from src.utils.propositional_logic import Atom, Iff, Or, Implies, And, Not


@dataclass
class Setup:
    attributes: list
    K: list
    concepts: dict

    def knowledge(self, target):
        if target in self.concepts:
            return self.K + [Iff(Atom(target), self.concepts[target])]
        return self.K

    def extend(self, V, target):
        V = dict(V)
        if target in self.concepts:
            V[target] = self.concepts[target].eval(V)
        return V

XTRAINSSetup = Setup(
    attributes=['passenger_car', 'freight_wagon', 'empty_wagon', 'reinforced_car',
                'open_roof_car', 'long_wagon', 'short_wagon', 'passenger_long_wagon',
                'two_passenger_cars', 'two_long_wagons', 'two_freight_wagons', 'three_wagons'],
    K=[
        Or(Atom('passenger_car'), Atom('freight_wagon'), Atom('empty_wagon')),
        Or(Atom('long_wagon'), Atom('short_wagon')),
        Implies(Atom('two_passenger_cars'), Atom('passenger_car')),
        Implies(Atom('two_long_wagons'), Atom('long_wagon')),
        Implies(Atom('two_freight_wagons'), Atom('freight_wagon')),
        Implies(Atom('passenger_long_wagon'), And(Atom('passenger_car'), Atom('long_wagon'))),
    ],
    concepts={
        'FreightWagon': Atom('freight_wagon'),
        'WarTrain': And(Atom('passenger_car'), Atom('reinforced_car')),
        'EmptyTrain': And(Atom('empty_wagon'), Not(Atom('passenger_car')),
                          Not(Atom('freight_wagon'))),
        'ReinforcedCar': Atom('reinforced_car'),
        'PassengerTrain': Or(Atom('passenger_long_wagon'), Atom('two_passenger_cars')),
        'LongTrain': Or(Atom('two_long_wagons'), Atom('three_wagons')),
        'FreightTrain': Atom('two_freight_wagons'),
        'LongWagon': Atom('long_wagon'),
        'RuralTrain': And(Atom('empty_wagon'), Or(Atom('passenger_car'), Atom('freight_wagon')),
                          Atom('short_wagon')),
        'MixedTrain': And(Atom('empty_wagon'), Atom('freight_wagon'), Atom('passenger_car')),
        'OpenRoofCar': Atom('open_roof_car'),
    },
)

GTSRBSetup = Setup(
    attributes=['Symbol', 'Black_Bar', 'Black_Symbol', 'Red_Ground',
                'Symbol_Speed80', 'Post', 'White_Ground'],
    K=[
        Implies(Atom('Symbol_Speed80'), Atom('Symbol')),
        Implies(Atom('Black_Symbol'), Atom('Symbol')),
        Implies(Atom('Symbol_Speed80'), Atom('Black_Symbol')),
        Not(And(Atom('Red_Ground'), Atom('White_Ground'))),
    ],
    concepts={
        'Symbol': Atom('Symbol'),
        'Black_Symbol': Atom('Black_Symbol'),
        'Symbol_Speed80': Atom('Symbol_Speed80'),
        'Red_Ground': Atom('Red_Ground'),
        'White_Ground': Atom('White_Ground'),
        'Black_Bar': Atom('Black_Bar'),
        'Post': Atom('Post'),
    },
)


CUBSetup = Setup(
    attributes=['attr_4', 'attr_5', 'attr_104', 'attr_112', 'attr_134', 'attr_148',
                'attr_198', 'attr_204', 'attr_306', 'attr_307'],
    K=[],
    concepts={
        'attr_4': Atom('attr_4'),
        'attr_5': Atom('attr_5'),
        'attr_104': Atom('attr_104'),
        'attr_112': Atom('attr_112'),
        'attr_134': Atom('attr_134'),
        'attr_148': Atom('attr_148'),
        'attr_198': Atom('attr_198'),
        'attr_204': Atom('attr_204'),
        'attr_306': Atom('attr_306'),
        'attr_307': Atom('attr_307'),
    },
)

CelebASetup = Setup(
    attributes=['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows',
                'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard',
                'Pale_Skin', 'Young'],
    K=[
        Not(And(Atom('Black_Hair'), Atom('Blond_Hair'))),
        Not(And(Atom('Black_Hair'), Atom('Brown_Hair'))),
        Not(And(Atom('Blond_Hair'), Atom('Brown_Hair'))),
        Not(And(Atom('Mustache'), Atom('No_Beard'))),
        Implies(Atom('Bald'), Not(Atom('Bangs'))),
        Implies(Atom('Bald'), Not(Or(Atom('Black_Hair'), Atom('Blond_Hair'),
                                     Atom('Brown_Hair')))),
        Implies(Atom('Mustache'), Atom('Male')),
        Implies(Not(Atom('No_Beard')), Atom('Male')),
    ],
    concepts={
        'Bald': Atom('Bald'),
        'Bangs': Atom('Bangs'),
        'Blond_Hair': Atom('Blond_Hair'),
        'Brown_Hair': Atom('Brown_Hair'),
        'Black_Hair': Atom('Black_Hair'),
        'Mustache': Atom('Mustache'),
        'Eyeglasses': Atom('Eyeglasses'),
        'Male': Atom('Male'),
        'Bushy_Eyebrows': Atom('Bushy_Eyebrows'),
    },
)


AdultSetup = Setup(
    attributes=['white', 'us_native', 'educ_ge13', 'prof_or_exec', 'married_civ', 'husband',
                'male', 'age_odd', 'age_30_55', 'hours_ge45', 'has_gain', 'has_loss'],
    K=[],
    concepts={
        'AdvancedCredential': And(Atom('educ_ge13'), Atom('prof_or_exec')),
        'MarriedMan': And(Atom('married_civ'), Atom('husband'), Atom('male')),
        'DemographicMajority': And(Atom('white'), Atom('us_native')),
        'InvestmentActivity': Or(Atom('has_gain'), Atom('has_loss')),
        'CareerPeak': And(Atom('age_30_55'), Atom('hours_ge45')),
        'OddAge': Atom('age_odd'),
    },
)
