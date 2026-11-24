import pickle

from matplotlib import pyplot as plt
from scipy.stats import truncnorm
from skimage.io import imread, imsave
import numpy as np
from os import listdir
import math

from src.generation.xtrains.concept_definitions import number_long_wagons, number_passenger_cars, number_freight_wagons, \
    has_empty_wagon, is_war_train, is_empty_train, is_passenger_train, is_long_train, is_freight_train, \
    is_long_freight_train, is_rural_train, is_mixed_train, is_type_a_train, is_type_b_train, is_type_c_train, \
    has_open_roof, has_car, has_roof, has_flat_roof, has_cargo, has_n_wheels_wagon, has_wheel_radius, cargo_quantity
from src.generation.xtrains.train_renderer import TrainRenderer


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean)/sd, (upp - mean)/sd, loc=mean, scale=sd)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def compress_images(path_to_images, output_path, images_per_npz=5_000):
    print("[INFO] Loading images...")
    image_names = listdir(path_to_images)

    for i, chunk in enumerate(chunks(image_names, images_per_npz)):
        images = []
        for img_path in chunk:
            # load the image and store it in the data list
            image = imread(path_to_images + img_path)
            images.append(image)
        print("[INFO] Processed {}/{}...".format(i + 1, math.ceil(len(image_names) / images_per_npz)))
        images = np.asarray(images)
        # save data list as npz
        np.savez_compressed(output_path + str(i).zfill(3), images=images)


def render(config, output_file):
    """Renders a train based on its configuration and saves the rendered image to output_file."""
    tr = TrainRenderer(config)
    img = tr.render_train()
    # imsave(PATH_TO_IMAGES + output_file, img)


def render_in_mem(config):
    """Renders a train based on its configuration and returns the img in memory."""
    tr = TrainRenderer(config)
    img = tr.render_train()
    return img

def render_in_mem_random(config):
    """Renders a train based on its configuration and returns the img in memory."""
    config['render_seed'] = config['render_seed'] + 1
    config['angle'] = (get_truncated_normal(mean=0, sd=0.3, low=-1, upp=1)
             .rvs(size=1, random_state=np.random.randint(0, 2 ** 31)) * 30)[0]
    tr = TrainRenderer(config)
    img = tr.render_train()
    return img


def filter_configs(configs, filenames):
    return {fname: configs[fname] for fname in filenames if fname in configs}


def show_img(img):
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def generate_config_file(filename, configs):
    """Generates a configuration file to store the trains' configurations."""
    with open(filename, 'wb') as config_file:
        pickle.dump(configs, config_file)
        config_file.close()


def generate_csv(path, trains):
    """Generates a csv file with some of the relevant features of the trains, based on their configurations."""
    file = open(path, 'w')

    file.write(
        'name,image_id,TypeA,TypeB,TypeC,WarTrain,EmptyTrain,PassengerTrain,LongFreightTrain,RuralTrain,MixedTrain,'
        'ReinforcedCar,PassengerCar,EmptyWagon,LongWagon,LongTrain,FreightTrain,FreightWagon,OpenRoofCar,HexagonCar,'
        'SuperellipseCar,TrapezoidCar,EllipseCar,BoxCar,SawRoof,TriangularRoof,FlatRoof,HexagonCargo,DiamondCargo,'
        'InvTriangleCargo,TriangleCargo,XLSquareCargo,MSquareCargo,SSquareCargo,TwoWheelsWagon,ThreeWheelsWagon,'
        'FourWheelsWagon,XSWheels,SWheels,MWheels,LWheels,CouplerHeight,WagonSpacing,Angle,TotalCargoQuantity,'
        'NumberOfWagons,NumberOfLongWagons,NumberOfFreightWagons,NumberOfPassengerCars\n')

    for i, train in enumerate(trains):
        name = str(i).zfill(9)
        image_id = name + ".png"
        number_of_long_wagons = number_long_wagons(train)
        number_of_passenger_cars = number_passenger_cars(train)
        number_of_freight_wagons = number_freight_wagons(train)

        long_wagon = int(number_of_long_wagons > 0)
        passenger_car = int(number_of_passenger_cars > 0)
        freight_wagon = int(number_of_freight_wagons > 0)
        empty_wagon = has_empty_wagon(train)

        war_train = is_war_train(train)
        empty_train = is_empty_train(train)
        passenger_train = is_passenger_train(train)
        long_train = is_long_train(train)
        freight_train = is_freight_train(train)
        long_freight_train = is_long_freight_train(train)
        rural_train = is_rural_train(train)
        mixed_train = is_mixed_train(train)
        type_a = is_type_a_train(train)
        type_b = is_type_b_train(train)
        type_c = is_type_c_train(train)

        roof_open = has_open_roof(train)

        car_reinforced = has_car(train, 'reinforced')

        hexagon_car = has_car(train, 'hexagon')
        superellipse_car = has_car(train, 'superellipse')
        trapezoid_car = has_car(train, 'trapezoid')
        ellipse_car = has_car(train, 'ellipse')
        box_car = has_car(train, 'box')

        saw_roof = has_roof(train, 'saw')
        triangular_roof = has_roof(train, 'triangular')
        flat_roof = has_flat_roof(train)

        hexagon_cargo = has_cargo(train, 'hexagon')
        diamond_cargo = has_cargo(train, 'diamond')
        inv_triangle_cargo = has_cargo(train, 'invtriangle')
        triangle_cargo = has_cargo(train, 'triangle')
        xl_square_cargo = has_cargo(train, 'xlsquare')
        m_square_cargo = has_cargo(train, 'msquare')
        s_square_cargo = has_cargo(train, 'ssquare')

        two_wheels_wagon = has_n_wheels_wagon(train, 2)
        three_wheels_wagon = has_n_wheels_wagon(train, 3)
        four_wheels_wagon = has_n_wheels_wagon(train, 4)

        xs_wheels = has_wheel_radius(train, 2)
        s_wheels = has_wheel_radius(train, 3)
        m_wheels = has_wheel_radius(train, 4)
        l_wheels = has_wheel_radius(train, 5)

        coupler_height = train['linker_y']
        wagon_spacing = train['half_space']
        angle = train['angle']
        total_cargo_quantity = cargo_quantity(train)

        number_of_wagons = len(train['cars'])
        number_of_long_wagons = number_long_wagons(train)
        number_of_freight_wagons = number_freight_wagons(train)
        number_of_passenger_cars = number_passenger_cars(train)

        file.write(
            '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'
            '{},{},{},{},{},{},{},{},{},{},{},{},{}'
            '\n'.format(
                name, image_id, type_a, type_b, type_c, war_train, empty_train, passenger_train, long_freight_train, rural_train,
                mixed_train, car_reinforced, passenger_car, empty_wagon, long_wagon, long_train, freight_train,
                freight_wagon, roof_open, hexagon_car, superellipse_car, trapezoid_car, ellipse_car, box_car, saw_roof,
                triangular_roof, flat_roof, hexagon_cargo, diamond_cargo, inv_triangle_cargo, triangle_cargo,
                xl_square_cargo, m_square_cargo, s_square_cargo, two_wheels_wagon, three_wheels_wagon,
                four_wheels_wagon, xs_wheels, s_wheels, m_wheels, l_wheels, coupler_height, wagon_spacing, angle,
                total_cargo_quantity, number_of_wagons, number_of_long_wagons, number_of_freight_wagons,
                number_of_passenger_cars))

    file.close()



"""
 Those are once implemented functions, that could be useful but are not being used right now
"""
def query_config_predicate(configs, predicate, value=1, max_samples=None):
    """
    Returns filtered configs using a predicate function that defines the concept.
    """
    results = []

    for config in configs:
        if predicate(config) == value:
            results.append(config)

        if max_samples is not None and len(results) >= max_samples:
            break

    return results

def query_config_not_predicate(configs, predicate, value=1, max_samples=None):
    """
    Returns filtered configs using a predicate function that defines the concept.
    """
    results = []

    for config in configs:
        if predicate(config) != value:
            results.append(config)

        if max_samples is not None and len(results) >= max_samples:
            break

    return results

def query_config_single(configs, key, value, max_samples=None):
    """
    Return configs that contain at least one car/shape/roof with the given value.
    """
    results = []

    for config in configs:
        for car in config['cars']:
            if car[1][key] == value:
                results.append(config)
                break  # avoid duplicates

        if max_samples is not None and len(results) >= max_samples:
            break

    return results