from dataclasses import dataclass
from pathlib import Path

from src.utils.common_paths import MAIN_NNS_PATH


@dataclass
class MainModelConfig:
    """ Base class for a main model configuration"""
    id: str
    dataset_id: str
    path: Path
    concepts: list
    layers: list
    protected: list

    def __str__(self):
        return self.id

TypeAConfig = MainModelConfig(
    id='type_a',
    dataset_id='xtrains',
    path=MAIN_NNS_PATH / "TypeA.pth",
    concepts=[
        "FreightWagon",
        "WarTrain", "EmptyTrain", "ReinforcedCar",
        "PassengerTrain", "LongTrain", "FreightTrain",
        "LongWagon", "RuralTrain", "MixedTrain",
        "OpenRoofCar",
    ],
    layers=[f'{i}' for i in [20,23,30,33,36]],
    protected=["PassengerCar", "FreightWagon", "ReinforcedCar",
               "EmptyWagon", "PassengerLongWagon", "notLongWagon",
               "notEmptyTrain"],
)

TypeBConfig = MainModelConfig(
    id='type_b',
    dataset_id='xtrains',
    path=MAIN_NNS_PATH / "TypeB.pth",
    concepts=[
        "FreightWagon",
        "WarTrain", "EmptyTrain", "ReinforcedCar",
        "PassengerTrain", "LongTrain", "FreightTrain",
        "LongWagon", "RuralTrain", "MixedTrain",
        "OpenRoofCar",
    ],
    layers=[f'{i}' for i in [20,23,30,33,]],
    protected=["PassengerCar", "FreightWagon", "ReinforcedCar",
               "EmptyWagon", "PassengerLongWagon", "notLongWagon",
               "notEmptyTrain"],
)


TypeCConfig = MainModelConfig(
    id='type_c',
    dataset_id='xtrains',
    path=MAIN_NNS_PATH / "TypeC.pth",
    concepts=[
        "FreightWagon",
        "WarTrain", "EmptyTrain", "ReinforcedCar",
        "PassengerTrain", "LongTrain", "FreightTrain",
        "LongWagon", "RuralTrain", "MixedTrain",
        "OpenRoofCar",
    ],
    layers=[f'{i}' for i in [20,23,30,33,36,39]],
    protected=["PassengerCar", "FreightWagon", "ReinforcedCar",
               "EmptyWagon", "PassengerLongWagon", "notLongWagon",
               "notEmptyTrain"],
)


ClassIDConfig = MainModelConfig(
    id='class_id',
    dataset_id='gtsrb',
    path=MAIN_NNS_PATH / 'ClassId.pth',
    concepts = ['Symbol', 'Black_Symbol', 'Symbol_Speed80',
                'Red_Ground', 'White_Ground', 'Black_Bar',
                'Post',],
    layers=['features.1.conv.2', 'features.5.conv.3',
            'features.9.conv.3', 'features.13.conv.3',
            'features.17.conv.3',],
    protected=['Symbol', 'Black_Symbol', 'Symbol_Speed80',
                'Red_Ground', 'White_Ground', 'Black_Bar',
                'Post',],
)


SpeciesConfig = MainModelConfig(
    id='species',
    dataset_id='cub',
    path=MAIN_NNS_PATH / 'Species.pth',
    concepts= ["attr_4", "attr_5", "attr_104", "attr_112",
              "attr_134","attr_148","attr_198" ,"attr_204",
              "attr_226" ,"attr_306" ,"attr_307"],
    layers=['layer4.0.bn3'],
    protected = ["attr_4", "attr_5", "attr_104", "attr_112",
                 "attr_134", "attr_148", "attr_198", "attr_204",
                 "attr_226", "attr_306", "attr_307"],
)


AttractiveConfig = MainModelConfig(
    id='attractive',
    dataset_id='celeba',
    path=MAIN_NNS_PATH / 'Attractive_resnet18.pth',
    concepts=['Bald', 'Bangs', 'Blond_Hair',
              'Brown_Hair', 'Black_Hair', 'Mustache',
              'Eyeglasses', 'Male', 'Bushy_Eyebrows',],
    layers=['layer4.0.bn2', 'layer4.1.bn2',],
    protected=['Bald', 'Bangs', 'Blond_Hair',
              'Brown_Hair', 'Black_Hair', 'Mustache',
              'Eyeglasses', 'Male', 'Bushy_Eyebrows',],
)


IncomeConfig = MainModelConfig(
    id='income',
    dataset_id='adult',
    path=MAIN_NNS_PATH / "Adult_income.pth",
    concepts=[
        "c_advanced_credential", "c_married_man",
        "c_demographic_majority", "c_investment_activity",
        "c_prime_working", "c_odd_age",
    ],
    layers=[f'{i}' for i in [0, 3]],
    protected=[
        "c_advanced_credential", "c_married_man",
        "c_demographic_majority", "c_investment_activity",
        "c_prime_working", "c_odd_age",
    ],
)