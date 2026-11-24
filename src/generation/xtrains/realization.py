import itertools
import random

from configs.dataset import XTRAINSConfig
from configs.experimental_setup import XTRAINSSetup
from src.framework.realization_cf import Realization
from src.generation.xtrains.config_generator import gen_cargo_settings
from src.generation.xtrains.utils import render_in_mem

FREIGHT = ['ssquare', 'msquare', 'xlsquare', 'triangle', 'invtriangle', 'diamond', 'hexagon']
VALUES = {
    'cargo_type': ['none', 'ssquare', 'msquare', 'circle', 'triangle', 'invtriangle',
                   'diamond', 'hexagon'],
    'shape': ['box', 'trapezoid', 'superellipse', 'reinforced'],
    'roof': ['open', 'flat', 'triangular', 'saw'],
    'width': [1.0, 1.42],
}


def circle(car):
    return car['cargo_type'] == 'circle'


def freight(car):
    return car['cargo_type'] in FREIGHT


def empty(car):
    return car['cargo_type'] == 'none'


def reinforced(car):
    return car['shape'] == 'reinforced'


def open_roof(car):
    return car['roof'] == 'open' and car['shape'] not in ('ellipse', 'hexagon')


def long(car):
    return car['width'] == 1.42


def short(car):
    return car['width'] != 1.42


def passenger_long(car):
    return circle(car) and long(car)


VALUATIONS = {
    'passenger_car': (circle, 1),
    'freight_wagon': (freight, 1),
    'empty_wagon': (empty, 1),
    'reinforced_car': (reinforced, 1),
    'open_roof_car': (open_roof, 1),
    'long_wagon': (long, 1),
    'short_wagon': (short, 1),
    'passenger_long_wagon': (passenger_long, 1),
    'two_passenger_cars': (circle, 2),
    'two_long_wagons': (long, 2),
    'two_freight_wagons': (freight, 2),
}


def valuation(cfg):
    cars = [car for _, car in cfg['cars']]
    V = {a: sum(pred(car) for car in cars) >= n for a, (pred, n) in VALUATIONS.items()}
    V['three_wagons'] = len(cars) >= 3
    return V


def attribute_columns(df, configs, attributes, name_col=XTRAINSConfig.img_path_col):
    df = df.copy()
    values = [valuation(configs[name]) for name in df[name_col]]
    for a in attributes:
        df[a] = [V[a] for V in values]
    return df


def apply(cfg, edits):
    """Copy of the config with the edits applied. An edit is (wagon, property, value);
    the property 'remove' removes the wagon."""
    # copy the config and each wagon, so the original config is never modified
    new = dict(cfg)
    cars = []
    for name, car in cfg['cars']:
        cars.append(dict(car))

    removed = []
    for wagon, prop, value in edits:
        # removals are applied at the end, so wagon indices stay valid meanwhile
        if prop == 'remove':
            removed.append(wagon)
            continue
        # an empty wagon that gets cargo needs a size and position for it
        if prop == 'cargo_type' and cars[wagon]['cargo_type'] == 'none':
            num, rad, edge, y = gen_cargo_settings(new, value, cars[wagon]['width'])
            cars[wagon]['cargo_num'] = num
            cars[wagon]['cargo_rad'] = rad
            cars[wagon]['cargo_edge'] = edge
            cars[wagon]['cargo_y'] = y
        cars[wagon][prop] = value

    # rebuild the wagon list without the removed wagons, renamed c0, c1, ...
    new['cars'] = []
    for wagon in range(len(cars)):
        if wagon not in removed:
            name = 'c' + str(len(new['cars']))
            new['cars'].append((name, cars[wagon]))
    return new



def possible_edits(cfg, changes):
    """Single edits that make one wagon start or stop satisfying the predicate
    of an attribute that S changed, in the direction S asked for."""
    edits = []
    for attribute, value in changes:
        for wagon in range(len(cfg['cars'])):
            car = cfg['cars'][wagon][1]

            # three_wagons can only become false, by removing a wagon
            if attribute == 'three_wagons':
                edit = (wagon, 'remove', None)
                if edit not in edits:
                    edits.append(edit)
                continue

            # the wagon predicate behind this attribute, e.g. circle for passenger_car
            pred = VALUATIONS[attribute][0]

            # this wagon already agrees with what S asked for: nothing to change here
            if pred(car) == value:
                continue

            # try every other value of every property of this wagon, one at a time,
            # and keep those that make the wagon agree with what S asked for
            for prop in VALUES:
                for v in VALUES[prop]:
                    if v == car[prop]:
                        continue
                    edited_car = dict(car)
                    edited_car[prop] = v
                    edit = (wagon, prop, v)
                    if pred(edited_car) == value and edit not in edits:
                        edits.append(edit)
    return edits

def same_property_twice(edit_set):
    """True if two edits change the same property of the same wagon."""
    # already tried at a smaller k, so it can be skipped
    seen = []
    for wagon, prop, value in edit_set:
        if (wagon, prop) in seen:
            return True
        seen.append((wagon, prop))
    return False


class XTRAINSRealization(Realization):
    def __init__(self, df, configs, setup=XTRAINSSetup, name_col=XTRAINSConfig.img_path_col,
                 max_edits=4, render=render_in_mem):
        self.df = df
        self.configs = configs
        self.name_col = name_col
        self.max_edits = max_edits
        self.render = render
        self.attributes = setup.attributes

    def config(self, x):
        return self.configs[self.df.loc[x, self.name_col]]

    def edited_config(self, x, V, Vcf):
        """The fewest edits whose valuation is exactly Vcf, ties broken by render_seed."""
        cfg = self.config(x)

        # what the counterfactual must look like, on the attributes of A only
        target = {}
        for a in self.attributes:
            target[a] = Vcf[a]

        # the single edits worth considering for the changes S asked for
        edits = possible_edits(cfg, self.changes(V, Vcf))

        # try sets of 1 edit, then 2, ... : the first k with a solution is the
        # minimal number of edits, so the counterfactual stays as close to x as possible
        for k in range(1, self.max_edits + 1):
            solutions = []
            for edit_set in itertools.combinations(edits, k):
                if same_property_twice(edit_set):
                    continue
                new = apply(cfg, edit_set)

                # soundness
                if valuation(new) == target:
                    solutions.append(new)

            # several solutions with k edits
            if solutions:
                return random.Random(cfg['render_seed']).choice(solutions)

        return None

    def realize(self, x, V, Vcf):
        new = self.edited_config(x, V, Vcf)
        if new is None:
            return None
        return self.render(new)


if __name__ == '__main__':
    import copy

    import matplotlib.pyplot as plt
    import pandas as pd

    from src.framework.counterfactual_intervention import counterfactual_intervention, labeling
    from src.utils.common_paths import DATASETS_PATH
    from src.utils.pickle import read_configs

    CONCEPT = 'WarTrain'

    configs = read_configs(XTRAINSConfig.configs_path, datasets_path=str(DATASETS_PATH), as_dict=True)
    exp_setup = XTRAINSSetup
    K = exp_setup.knowledge(CONCEPT)

    df = attribute_columns(pd.read_csv(XTRAINSConfig.labels_path), configs, exp_setup.attributes)
    R = XTRAINSRealization(df, configs)
    phi = exp_setup.concepts[CONCEPT]
    xs = [x for x in df.index if phi.eval(labeling(df, x, exp_setup.attributes))][:10]
    for x in xs:
        V = exp_setup.extend(labeling(df, x, exp_setup.attributes), CONCEPT)
        x_cf = counterfactual_intervention(x, V, CONCEPT, K, R, seed=x)
        if x_cf is None:
            continue
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(render_in_mem(copy.deepcopy(R.config(x))))
        ax[0].set_title('before')
        ax[1].imshow(x_cf)
        ax[1].set_title('after')
        for axis in ax:
            axis.axis('off')
        fig.suptitle(f'{CONCEPT}, {df.loc[x, R.name_col]}')
        plt.show()