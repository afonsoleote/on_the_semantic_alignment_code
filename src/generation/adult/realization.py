from configs.experimental_setup import AdultSetup
from src.framework.realization_cf import Realization

ENCODINGS = {
    'workclass': ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
                  'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'],
    'marital-status': ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated',
                       'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'],
    'occupation': ['Tech-support', 'Craft-repair', 'Other-service', 'Sales',
                   'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
                   'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
                   'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'],
    'relationship': ['Wife', 'Own-child', 'Husband', 'Not-in-family',
                     'Other-relative', 'Unmarried'],
    'race': ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'],
    'sex': ['Female', 'Male'],
    'native-country': ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada',
                       'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece',
                       'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy',
                       'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland',
                       'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti',
                       'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand',
                       'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong',
                       'Holand-Netherlands'],
}
CODE = {c: {n: i for i, n in enumerate(names)} for c, names in ENCODINGS.items()}


def attribute_columns(df):
    df = df.copy()
    df['white'] = df['race'] == CODE['race']['White']
    df['us_native'] = df['native-country'] == CODE['native-country']['United-States']
    df['educ_ge13'] = df['education-num'] >= 13
    df['prof_or_exec'] = df['occupation'].isin([CODE['occupation']['Prof-specialty'],
                                                CODE['occupation']['Exec-managerial']])
    df['married_civ'] = df['marital-status'] == CODE['marital-status']['Married-civ-spouse']
    df['husband'] = df['relationship'] == CODE['relationship']['Husband']
    df['male'] = df['sex'] == CODE['sex']['Male']
    df['age_odd'] = df['age'] % 2 == 1
    df['age_30_55'] = df['age'].between(30, 55)
    df['hours_ge45'] = df['hours-per-week'] >= 45
    df['has_gain'] = df['capital-gain'] > 0
    df['has_loss'] = df['capital-loss'] > 0
    return df


class AdultRealization(Realization):
    EDITS = {
        ('white', True): ('race', CODE['race']['White']),
        ('us_native', True): ('native-country', CODE['native-country']['United-States']),
        ('educ_ge13', True): ('education-num', 13),
        ('prof_or_exec', True): ('occupation', CODE['occupation']['Prof-specialty']),
        ('married_civ', True): ('marital-status', CODE['marital-status']['Married-civ-spouse']),
        ('husband', True): ('relationship', CODE['relationship']['Husband']),
        ('male', True): ('sex', CODE['sex']['Male']),
        ('age_odd', True): ('age', None),
        ('age_30_55', True): ('age', 45),
        ('hours_ge45', True): ('hours-per-week', 50),
        ('has_gain', False): ('capital-gain', 0),
        ('has_loss', False): ('capital-loss', 0),
    }

    def __init__(self, df, setup=AdultSetup):
        self.df = df
        self.attributes = setup.attributes

    def realize(self, x, V, Vcf):
        row = self.df.loc[x].copy()
        for a, value in self.changes(V, Vcf):
            if (a, value) not in self.EDITS:
                return None
            feature, new = self.EDITS[(a, value)]
            row[feature] = row[feature] + 1 if new is None else new
        return row


if __name__ == '__main__':
    import pandas as pd

    from src.framework.counterfactual_intervention import counterfactual_intervention, labeling
    from src.utils.common_paths import ADULT_CSV_PATH

    CONCEPT = 'DemographicMajority'

    df = attribute_columns(pd.read_csv(ADULT_CSV_PATH))
    exp_setup = AdultSetup
    K = exp_setup.knowledge(CONCEPT)
    f = exp_setup.concepts[CONCEPT]

    R = AdultRealization(df)
    xs = [x for x in df.index if f.eval(labeling(df, x, exp_setup.attributes)) == False][:10]
    for x in xs:
        V = exp_setup.extend(labeling(df, x, exp_setup.attributes), CONCEPT)

        x_cf = counterfactual_intervention(x, V, CONCEPT, K, R)
        print(f'concept {CONCEPT}, value before: {f.eval(V)}')
        if x_cf is None:
            print('not realizable for this row')
        else:
            for feature in ENCODINGS.keys() | {'age', 'education-num', 'hours-per-week',
                                               'capital-gain', 'capital-loss'}:
                if df.loc[x, feature] != x_cf[feature]:
                    print(f'  {feature}: {df.loc[x, feature]} -> {x_cf[feature]}')