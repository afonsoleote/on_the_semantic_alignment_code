import numpy as np

from src.generation.xtrains.utils import get_truncated_normal

SHAPES = ['box', 'ellipse', 'trapezoid', 'superellipse', 'hexagon', 'reinforced']
WIDTHS = [1, 1.42]
WHEELS = [2, 3, 4]
WHEEL_RADS = [2, 3, 4, 5]
C_TYPES = ['none', 'ssquare', 'msquare', 'xlsquare', 'circle', 'triangle', 'invtriangle', 'diamond', 'hexagon']
C_NUM = [1, 2, 3]
C_RADS = [2, 3, 4, 5]
C_YS = [6, 7, 8, 9, 10]
LINKER_YS = [13, 14]
HALF_SPACES = [1, 2]
ROOFS = ['open', 'flat', 'triangular', 'saw']
RENDER = False


def gen_cargo_settings(train, cargo_type, width):
    """Generates the configuration of a wagon's cargo."""
    cargo_num, cargo_rad, cargo_edge, cargo_y = (0, 0, 0, 0)

    if width > 1:  # long train
        if cargo_type == 'msquare':
            cargo_num = np.random.choice(C_NUM, p=[0.65, 0.35, 0.0])
            cargo_rad = 2
            cargo_edge = 10
        elif cargo_type == 'xlsquare':
            cargo_num = 1
            cargo_rad = 2
            cargo_edge = 9
        elif cargo_type == 'hexagon':
            cargo_num = np.random.choice(C_NUM, p=[0.65, 0.35, 0.0])
            if cargo_num == 2:
                cargo_rad = 2
                cargo_edge = np.random.choice([10, 11])
            else:  # only one hexagonal cargo
                cargo_rad = np.random.choice(C_RADS, p=[0.3, 0.4, 0.3, 0.0])
                cargo_edge = 9

        else:  # any other cargo shape
            cargo_rad = np.random.choice(C_RADS, p=[0.27, 0.3, 0.38, 0.05])
            if cargo_rad == 2:
                cargo_num = np.random.choice(C_NUM, p=[0.3, 0.3, 0.4])
                if cargo_num == 3:
                    cargo_edge = np.random.choice([9, 10])
                else:
                    cargo_edge = np.random.choice([8, 9, 10, 11, 12])
            elif cargo_rad == 3:
                cargo_num = np.random.choice(C_NUM, p=[0.55, 0.35, 0.1])
                if cargo_num == 3:
                    cargo_edge = 8
                else:
                    cargo_edge = np.random.choice([8, 9, 10])
            elif cargo_rad == 4:
                cargo_num = np.random.choice(C_NUM, p=[0.6, 0.4, 0.0])
                cargo_edge = 8
            elif cargo_rad == 5:
                cargo_num = 1
                cargo_edge = 8

    else:  # short train
        if cargo_type == 'msquare':
            cargo_num = 1
            cargo_rad = 2
            cargo_edge = 9
        elif cargo_type == 'hexagon':
            cargo_num = 1
            cargo_rad = np.random.choice(C_RADS, p=[0.8, 0.2, 0.0, 0.0])
            cargo_edge = 9
        else:  # any other cargo shape
            cargo_rad = np.random.choice(C_RADS, p=[0.4, 0.3, 0.3, 0.0])
            if cargo_rad == 2:
                cargo_num = np.random.choice(C_NUM, p=[0.8, 0.2, 0.0])
                if cargo_num == 2:
                    cargo_edge = 7
                else:
                    cargo_edge = 9
            else:  # cargo rad 3, 4 or 5
                cargo_num = 1
                cargo_edge = 9

    if cargo_rad == 2:
        cargo_y = np.random.choice(C_YS, p=[0.05, 0.2, 0.3, 0.25, 0.2])
    elif cargo_rad == 3:
        cargo_y = np.random.choice(C_YS, p=[0.0, 0.2, 0.3, 0.3, 0.2])
    elif cargo_rad == 4:
        cargo_y = np.random.choice(C_YS, p=[0.0, 0.0, 0.5, 0.5, 0.0])

    if (cargo_rad == 5) or (train['wheel_radius'] == 5):
        cargo_y = 8

    if (train['wheel_radius'] == 5) and (cargo_rad == 5):
        train['wheel_radius'] = 4
        train['wheel_base'] = train['wheel_base'] + 1

    return cargo_num, cargo_rad, cargo_edge, cargo_y


def generate_car(train):
    """Generates the configuration of a train's wagon."""
    car = {}
    wheel_radius = train['wheel_radius']
    width = np.random.choice(WIDTHS, p=[2/3, 1/3])
    cargo_type = 'none'
    wheels = 2
    roof = 'open'

    if width > 1:  # long train
        if wheel_radius == 2:
            wheels = np.random.choice(WHEELS, p=[0.0, 0.8, 0.2])
        elif wheel_radius == 3:
            wheels = np.random.choice(WHEELS, p=[0.5, 0.4, 0.1])
        elif wheel_radius == 4:
            wheels = np.random.choice(WHEELS, p=[0.5, 0.5, 0.0])
        elif wheel_radius == 5:
            wheels = np.random.choice(WHEELS, p=[0.5, 0.5, 0.0])

        shape = np.random.choice(SHAPES, p=[0.1, 0.2, 0.1, 0.2, 0.2, 0.2])

        if shape == 'box':
            roof = np.random.choice(ROOFS, p=[0.2, 0.3, 0.2, 0.3])
            if roof == 'open':
                cargo_type = np.random.choice(C_TYPES, p=[0.14, 0.12, 0.12, 0.11, 0.1, 0.1, 0.1, 0.1, 0.11])
            elif roof == 'flat':
                cargo_type = np.random.choice(C_TYPES, p=[0.1, 0.12, 0.12, 0.11, 0.14, 0.1, 0.1, 0.1, 0.11])
            elif roof == 'triangular':
                cargo_type = np.random.choice(C_TYPES, p=[0.1, 0.11, 0.11, 0.1, 0.16, 0.11, 0.11, 0.11, 0.09])
            elif roof == 'saw':
                cargo_type = np.random.choice(C_TYPES, p=[0.1, 0.11, 0.11, 0.1, 0.16, 0.1, 0.1, 0.1, 0.12])

        elif shape == 'ellipse':
            roof = 'open'
            cargo_type = np.random.choice(C_TYPES, p=[0.1, 0.11, 0.1, 0.1, 0.17, 0.11, 0.11, 0.1, 0.1])
            if wheel_radius == 2:
                train['wheel_base'] = train['wheel_base'] - 1
                train['wheel_radius'] = 3

        elif shape == 'trapezoid':
            roof = np.random.choice(ROOFS, p=[0.4, 0.2, 0.2, 0.2])
            if roof == 'open':
                cargo_type = np.random.choice(C_TYPES, p=[0.15, 0.11, 0.1, 0.1, 0.1, 0.11, 0.11, 0.11, 0.11])
            elif roof == 'flat':
                cargo_type = np.random.choice(C_TYPES, p=[0.1, 0.11, 0.11, 0.11, 0.13, 0.11, 0.11, 0.11, 0.11])
            elif roof == 'triangular':
                cargo_type = np.random.choice(C_TYPES, p=[0.13, 0.1, 0.1, 0.1, 0.14, 0.11, 0.11, 0.11, 0.1])
            elif roof == 'saw':
                cargo_type = np.random.choice(C_TYPES, p=[0.11, 0.1, 0.1, 0.1, 0.14, 0.11, 0.11, 0.11, 0.12])

        elif shape == 'superellipse':
            roof = np.random.choice(ROOFS, p=[0.55, 0.15, 0.15, 0.15])
            if roof == 'open':
                cargo_type = np.random.choice(C_TYPES, p=[0.2, 0.09, 0.09, 0.09, 0.13, 0.1, 0.1, 0.1, 0.1])
            elif roof == 'flat':
                cargo_type = np.random.choice(C_TYPES, p=[0.14, 0.09, 0.09, 0.09, 0.15, 0.11, 0.11, 0.11, 0.11])
            elif roof == 'triangular':
                cargo_type = np.random.choice(C_TYPES, p=[0.14, 0.09, 0.09, 0.09, 0.16, 0.11, 0.11, 0.11, 0.1])
            elif roof == 'saw':
                cargo_type = np.random.choice(C_TYPES, p=[0.12, 0.09, 0.09, 0.09, 0.16, 0.11, 0.11, 0.11, 0.12])

        elif shape == 'hexagon':
            roof = 'open'
            cargo_type = np.random.choice(C_TYPES, p=[0.11, 0.11, 0.1, 0.1, 0.1, 0.12, 0.12, 0.12, 0.12])

        elif shape == 'reinforced':
            roof = np.random.choice(ROOFS, p=[0.25, 0.25, 0.25, 0.25])
            if roof == 'open':
                cargo_type = np.random.choice(C_TYPES, p=[0.14, 0.12, 0.12, 0.11, 0.1, 0.1, 0.1, 0.1, 0.11])
            elif roof == 'flat':
                cargo_type = np.random.choice(C_TYPES, p=[0.1, 0.12, 0.12, 0.11, 0.14, 0.1, 0.1, 0.1, 0.11])
            elif roof == 'triangular':
                cargo_type = np.random.choice(C_TYPES, p=[0.1, 0.11, 0.11, 0.1, 0.16, 0.11, 0.11, 0.11, 0.09])
            elif roof == 'saw':
                cargo_type = np.random.choice(C_TYPES, p=[0.1, 0.11, 0.11, 0.1, 0.16, 0.1, 0.1, 0.1, 0.12])

    else:  # short train
        if wheel_radius == 2:
            wheels = np.random.choice(WHEELS, p=[0.1, 0.9, 0.0])
        elif wheel_radius == 3:
            wheels = np.random.choice(WHEELS, p=[0.9, 0.1, 0.0])
        elif wheel_radius == 4:
            wheels = 2
        elif wheel_radius == 5:
            wheels = 2

        shape = np.random.choice(SHAPES, p=[0.2, 0.1, 0.2, 0.2, 0.1, 0.2])

        if shape == 'box':
            roof = np.random.choice(ROOFS, p=[0.2, 0.3, 0.2, 0.3])
            if roof == 'open':
                cargo_type = np.random.choice(C_TYPES, p=[0.2, 0.14, 0.11, 0.0, 0.11, 0.11, 0.11, 0.11, 0.11])
            elif roof == 'flat':
                cargo_type = np.random.choice(C_TYPES, p=[0.2, 0.13, 0.11, 0.0, 0.15, 0.1, 0.1, 0.1, 0.11])
            elif roof == 'triangular':
                cargo_type = np.random.choice(C_TYPES, p=[0.13, 0.13, 0.1, 0.0, 0.19, 0.12, 0.12, 0.11, 0.1])
            elif roof == 'saw':
                cargo_type = np.random.choice(C_TYPES, p=[0.14, 0.13, 0.1, 0.0, 0.19, 0.11, 0.11, 0.1, 0.12])

        elif shape == 'ellipse':
            roof = 'open'
            cargo_type = np.random.choice(C_TYPES, p=[0.14, 0.12, 0.11, 0.0, 0.19, 0.11, 0.11, 0.11, 0.11])
            if wheel_radius == 2:
                train['wheel_base'] = train['wheel_base'] - 1
                train['wheel_radius'] = 3

        elif shape == 'trapezoid':
            roof = np.random.choice(ROOFS, p=[0.4, 0.2, 0.2, 0.2])
            if roof == 'open':
                cargo_type = np.random.choice(C_TYPES, p=[0.2, 0.12, 0.1, 0.0, 0.1, 0.12, 0.12, 0.12, 0.12])
            elif roof == 'flat':
                cargo_type = np.random.choice(C_TYPES, p=[0.15, 0.12, 0.12, 0.0, 0.15, 0.12, 0.11, 0.11, 0.12])
            elif roof == 'triangular':
                cargo_type = np.random.choice(C_TYPES, p=[0.15, 0.11, 0.11, 0.0, 0.16, 0.12, 0.12, 0.12, 0.11])
            elif roof == 'saw':
                cargo_type = np.random.choice(C_TYPES, p=[0.14, 0.11, 0.11, 0.0, 0.15, 0.12, 0.12, 0.12, 0.13])

        elif shape == 'superellipse':
            roof = np.random.choice(ROOFS, p=[0.55, 0.15, 0.15, 0.15])
            if roof == 'open':
                cargo_type = np.random.choice(C_TYPES, p=[0.23, 0.1, 0.1, 0.0, 0.13, 0.11, 0.11, 0.11, 0.11])
            elif roof == 'flat':
                cargo_type = np.random.choice(C_TYPES, p=[0.18, 0.1, 0.1, 0.0, 0.16, 0.12, 0.11, 0.11, 0.12])
            elif roof == 'triangular':
                cargo_type = np.random.choice(C_TYPES, p=[0.17, 0.1, 0.1, 0.0, 0.18, 0.12, 0.11, 0.11, 0.11])
            elif roof == 'saw':
                cargo_type = np.random.choice(C_TYPES, p=[0.16, 0.12, 0.1, 0.0, 0.17, 0.11, 0.11, 0.11, 0.12])

        elif shape == 'hexagon':
            roof = 'open'
            cargo_type = np.random.choice(C_TYPES, p=[0.15, 0.13, 0.1, 0.0, 0.1, 0.13, 0.13, 0.13, 0.13])

        elif shape == 'reinforced':
            roof = np.random.choice(ROOFS, p=[0.25, 0.25, 0.25, 0.25])
            if roof == 'open':
                cargo_type = np.random.choice(C_TYPES, p=[0.19, 0.15, 0.11, 0.0, 0.11, 0.11, 0.11, 0.11, 0.11])
            elif roof == 'flat':
                cargo_type = np.random.choice(C_TYPES, p=[0.15, 0.15, 0.12, 0.0, 0.14, 0.11, 0.11, 0.11, 0.11])
            elif roof == 'triangular':
                cargo_type = np.random.choice(C_TYPES, p=[0.13, 0.14, 0.11, 0.0, 0.19, 0.11, 0.11, 0.11, 0.1])
            elif roof == 'saw':
                cargo_type = np.random.choice(C_TYPES, p=[0.14, 0.14, 0.11, 0.0, 0.19, 0.1, 0.1, 0.1, 0.12])

    car['shape'] = shape
    car['height'] = 1
    car['width'] = width
    car['wheels'] = wheels
    car['cargo_type'] = cargo_type
    car['roof'] = roof
    car['img_shape'] = (int(train['base_height'] * car['height']), int(train['base_width'] * car['width']), 3)

    if cargo_type != 'none':
        (cargo_num, cargo_rad, cargo_edge, cargo_y) = gen_cargo_settings(train, cargo_type, width)

        car['cargo_num'] = cargo_num
        car['cargo_edge'] = cargo_edge
        car['cargo_rad'] = cargo_rad
        car['cargo_y'] = cargo_y
    else:
        car['cargo_num'] = 0

    return car


def generate_train_config(train_id, backgrounds):
    """Generates the configuration of a train."""
    train = {}

    wheel_radius = np.random.choice(WHEEL_RADS, p=[0.1, 0.45, 0.4, 0.05])

    train['id'] = train_id
    train['base_width'] = 24
    train['base_height'] = 20
    train['roof_height'] = 3
    train['wheel_base'] = train['base_height'] - train['roof_height'] - wheel_radius - 1
    train['wheel_radius'] = wheel_radius
    train['linker_y'] = np.random.choice(LINKER_YS)
    train['half_space'] = np.random.choice(HALF_SPACES)
    train['background'] = np.random.choice(backgrounds)
    train['angle'] = (get_truncated_normal(mean=0, sd=0.3, low=-1, upp=1)
             .rvs(size=1, random_state=np.random.randint(0, 2 ** 31)) * 30)[0]

    # num_cars varies between 1 and 4
    dist = get_truncated_normal(mean=0, sd=0.5, low=-1, upp=1)
    num_cars = dist.rvs(size=1, random_state=np.random.randint(0, 2 ** 31))
    num_cars = num_cars * 2 + 3
    num_cars = num_cars.astype(int)
    num_cars = num_cars[0]

    cars = []
    for i in range(0, num_cars):
        car = generate_car(train)
        cars.append(('c{}'.format(i), car))
    train['cars'] = cars

    return train

