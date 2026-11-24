def number_long_wagons(train):
    res = 0
    for car in train['cars']:
        if car[1]['width'] > 1:
            res += 1
    return res


def has_long_wagon(train):
    for car in train['cars']:
        if car[1]['width'] > 1:
            return 1
    return 0


def number_passenger_cars(train):
    res = 0
    for car in train['cars']:
        if car[1]['cargo_type'] == 'circle':
            res += 1
    return res


def has_passenger_car(train):
    for car in train['cars']:
        if car[1]['cargo_type'] == 'circle':
            return 1
    return 0


def number_freight_wagons(train):
    res = 0
    for car in train['cars']:
        if car[1]['cargo_type'] != 'circle' and car[1]['cargo_type'] != 'none':
            res += 1
    return res


def has_freight_wagon(train):
    for car in train['cars']:
        if car[1]['cargo_type'] != 'circle' and car[1]['cargo_type'] != 'none':
            return 1
    return 0


def has_empty_wagon(train):
    for car in train['cars']:
        if car[1]['cargo_type'] == 'none':
            return 1
    return 0


def is_long_train(train):
    res = 0
    length = len(train['cars'])

    if length >= 3:
        res = 1
        return res

    count_long = 0
    for car in train['cars']:
        if car[1]['width'] > 1:
            count_long = count_long + 1

    if count_long >= 2:
        res = 1
    return res


def is_war_train(train):
    for car in train['cars']:
        if car[1]['shape'] == 'reinforced':
            return has_passenger_car(train)
    return 0


def is_passenger_train(train):
    count = 0
    for car in train['cars']:
        if car[1]['cargo_type'] == 'circle':
            count = count + 1
            if car[1]['width'] > 1:
                return 1
            if count >= 2:
                return 1
    return 0


def is_freight_train(train):
    count = 0
    for car in train['cars']:
        if car[1]['cargo_type'] != 'circle' and car[1]['cargo_type'] != 'none':
            count = count + 1
            if count >= 2:
                return 1
    return 0


def is_long_freight_train(train):
    if is_long_train(train) and is_freight_train(train):
        return 1
    else:
        return 0


def is_empty_train(train):
    for car in train['cars']:
        if car[1]['cargo_type'] != 'none':
            return 0
    return 1


def is_type_a_train(train):
    if is_war_train(train) or is_empty_train(train):
        return 1
    else:
        return 0


def is_type_b_train(train):
    if is_passenger_train(train) or is_long_freight_train(train):
        return 1
    else:
        return 0


def is_rural_train(train):
    if has_empty_wagon(train)\
            and (has_passenger_car(train) or has_freight_wagon(train))\
            and not has_long_wagon(train):
        return 1
    else:
        return 0


def is_mixed_train(train):
    if has_freight_wagon(train) and has_passenger_car(train) and has_empty_wagon(train):
        return 1
    else:
        return 0


def is_type_c_train(train):
    if is_rural_train(train) or is_mixed_train(train):
        return 1
    else:
        return 0


def has_roof(train, roof):
    for car in train['cars']:
        if car[1]['roof'] == roof:
            return 1
    return 0


def has_flat_roof(train):
    """Note: Hexagon wagons have no additional roof, but do have a 'flat roof'."""
    for car in train['cars']:
        if car[1]['roof'] == 'flat' or car[1]['shape'] == 'hexagon':
            return 1
    return 0


def has_open_roof(train):
    """Note: Ellipse and hexagon wagons have no additional roof, but are not 'open roof'."""
    for car in train['cars']:
        if car[1]['roof'] == 'open' and car[1]['shape'] != 'ellipse' and car[1]['shape'] != 'hexagon':
            return 1
    return 0


def has_car(train, shape):
    for car in train['cars']:
        if car[1]['shape'] == shape:
            return 1
    return 0

def has_reinforced_car(train):
    for car in train['cars']:
        if car[1]['shape'] == 'reinforced':
            return 1
    return 0


def has_cargo(train, cargo):
    for car in train['cars']:
        if car[1]['cargo_type'] == cargo:
            return 1
    return 0


def has_n_wheels_wagon(train, n):
    for car in train['cars']:
        if car[1]['wheels'] == n:
            return 1
    return 0


def has_wheel_radius(train, radius):
    return int(train['wheel_radius'] == radius)


def cargo_quantity(train):
    count = 0
    for car in train['cars']:
        count += car[1]['cargo_num']
    return count

def is_passenger_long_wagon(car):
    if car[1]['cargo_type'] == 'circle' and car[1]['width'] > 1:
        return 1
    return 0

def has_passenger_long_wagon(train):
    for car in train['cars']:
        if is_passenger_long_wagon(car):
            return 1
    return 0

def has_2_long_wagon(train):
    if number_long_wagons(train) >= 2:
        return 1
    return 0

def has_3_wagon(train):
    if len(train['cars']) >= 3:
        return 1
    return 0

def has_2_passenger_car(train):
    if number_passenger_cars(train) >= 2:
        return 1
    return 0

def has_2_freight_wagon(train):
    if number_freight_wagons(train) >= 2:
        return 1
    return 0