from sklearn.linear_model import LogisticRegression

from src.probing.probes.lgbm_probe import LGBMProbe
from src.probing.probes.logistic_probe import LogisticProbe
from src.probing.probes.nn_probe import NNProbe
from src.probing.probes.ridge_probe import RidgeProbe
from src.utils.models.architectures import (LargeDenseModel, MediumDenseModel,
                                            LinearModel, StandardDenseModel)

ARCH_REGISTRY = {
    LinearModel.id: LinearModel,
    StandardDenseModel.id: StandardDenseModel,
    MediumDenseModel.id: MediumDenseModel,
    LargeDenseModel.id: LargeDenseModel,
}

def train_model(dm, config):
    if config['probe'] == 'linear':
        return train_linear(dm, config)

    elif config['probe'] == 'ridge':
        return train_ridge(dm, config)

    elif config['probe'] == 'lgbm':
        return train_lgbm_tree(dm, config)

    elif config['probe'] in ARCH_REGISTRY:
        return train_nn(dm, config)

    elif config['probe'] == 'linear_cf':
        return train_linear_cf(dm, config)


def train_linear(dm, config):
    train_config = {}

    print("[INFO] Training model.")
    x_tr, y_tr = dm.get_x_y('train')
    x_tr = x_tr.cpu().numpy()
    y_tr = y_tr.cpu().numpy().reshape(-1)
    print(x_tr.shape)

    x_val, y_val = dm.get_x_y('validate')
    x_val = x_val.cpu().numpy()
    y_val = y_val.cpu().numpy().reshape(-1)

    probe = LogisticProbe(seed=config["seed"])
    probe.fit(x_tr, y_tr)

    print("[INFO] Evaluating train set performance.")
    train_config["tr_acc"] = probe.score(x_tr, y_tr)
    print(f"[INFO] Train accuracy: {train_config['tr_acc']:.2%}")

    print("[INFO] Evaluating validation set performance.")
    train_config["val_acc"] = probe.score(x_val, y_val)
    print(f"[INFO] Validation accuracy: {train_config['val_acc']:.2%}")

    x_test, y_test = dm.get_x_y('test')
    x_test = x_test.cpu().numpy()
    y_test = y_test.cpu().numpy().reshape(-1)

    print("[INFO] Evaluating test set performance.")
    train_config["test_acc"] = probe.score(x_test, y_test)
    print(f"[INFO] Test accuracy: {train_config['test_acc']:.2%}")

    return probe, train_config


def train_lgbm_tree(dm, config):
    train_config = {}

    print("[INFO] Training model.")

    x_tr, y_tr = dm.get_x_y('train')
    x_tr = x_tr.cpu().numpy()
    y_tr = y_tr.cpu().numpy().reshape(-1)

    x_val, y_val = dm.get_x_y('validate')
    x_val = x_val.cpu().numpy()
    y_val = y_val.cpu().numpy().reshape(-1)

    probe = LGBMProbe(config['seed'])
    probe.fit(x_tr, y_tr, x_val, y_val)

    print("[INFO] Evaluating train set performance.")
    train_config["tr_acc"] = probe.score(x_tr, y_tr)
    print(f"[INFO] Train accuracy: {train_config['tr_acc']:.2%}")

    print("[INFO] Evaluating validation set performance.")
    train_config["val_acc"] = probe.score(x_val, y_val)
    print(f"[INFO] Val accuracy: {train_config['val_acc']:.2%}")

    print("[INFO] Evaluating test set performance.")

    x_test, y_test = dm.get_x_y('test')
    x_test = x_test.cpu().numpy()
    y_test = y_test.cpu().numpy().reshape(-1)

    train_config["test_acc"] = probe.score(x_test, y_test)
    print(f"[INFO] Test accuracy: {train_config['test_acc']:.2%}")

    return probe, train_config


def train_ridge(dm, config):
    train_config = {}

    print("[INFO] Training model.")

    x_tr, y_tr = dm.get_x_y('train')
    x_tr = x_tr.cpu().numpy()
    y_tr = y_tr.cpu().numpy().reshape((-1,))

    x_val, y_val = dm.get_x_y('validate')
    x_val = x_val.cpu().numpy()
    y_val = y_val.cpu().numpy().reshape((-1,))

    alpha_grid = [0.01] # , 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]

    probe = RidgeProbe(config['seed'])
    probe.fit(
        x_tr,
        y_tr,
        x_val,
        y_val,
        alpha_grid=alpha_grid
    )

    print(f"[INFO] Best alpha: {probe.alpha}")

    print("[INFO] Evaluating train set performance.")
    train_config['tr_acc'] = (probe.predict(x_tr) == y_tr).mean()
    print(f"[INFO] Train accuracy: {train_config['tr_acc']:.2%}")

    print("[INFO] Evaluating validation set performance.")
    train_config['val_acc'] = (probe.predict(x_val) == y_val).mean()
    print(f"[INFO] Val accuracy: {train_config['val_acc']:.2%}")

    print("[INFO] Evaluating test set performance.")
    x_test, y_test = dm.get_x_y('test')
    x_test = x_test.cpu().numpy()
    y_test = y_test.cpu().numpy().reshape((-1,))

    train_config['test_acc'] = (probe.predict(x_test) == y_test).mean()
    print(f"[INFO] Test accuracy: {train_config['test_acc']:.2%}")

    return probe, train_config


def train_nn(dm, config):

    train_config = {}

    print("[INFO] Training model.")

    x_tr, y_tr = dm.get_x_y('train')
    x_tr = x_tr.cpu().numpy()
    y_tr = y_tr.cpu().numpy().reshape(-1)

    x_val, y_val = dm.get_x_y('validate')
    x_val = x_val.cpu().numpy()
    y_val = y_val.cpu().numpy().reshape(-1)

    probe = NNProbe(architecture=ARCH_REGISTRY[config["probe"]], seed=config["seed"])

    kwargs = {"verbose": False, "early_stopping": False}
    if "epochs" in config:
        kwargs["epochs"] = config["epochs"]

    _, tr_loss, val_loss = probe.fit(x_tr, y_tr, x_val, y_val, **kwargs)

    print("[INFO] Evaluating train set performance.")
    tr_acc = (probe.predict(x_tr) == y_tr).mean()
    train_config["tr_acc"] = tr_acc
    print(f"[INFO] Train accuracy: {train_config['tr_acc']:.2%}")

    print("[INFO] Evaluating validation set performance.")
    val_acc = (probe.predict(x_val) == y_val).mean()
    train_config["val_acc"] = val_acc
    print(f"[INFO] Val accuracy: {train_config['val_acc']:.2%}")

    print("[INFO] Evaluating test set performance.")
    x_test, y_test = dm.get_x_y('test')
    x_test = x_test.cpu().numpy()
    y_test = y_test.cpu().numpy().reshape(-1)

    test_acc = (probe.predict(x_test) == y_test).mean()
    train_config["test_acc"] = test_acc

    print(f"[INFO] Test accuracy: {train_config['test_acc']:.2%}")

    train_config["tr_loss"] = tr_loss
    train_config["val_loss"] = val_loss
    print(f"[INFO] Train loss: {train_config['tr_loss']}")
    print(f"[INFO] Validation loss: {train_config['val_loss']}")

    return probe, train_config

def train_linear_cf(dm, config):
    train_config = {}

    print("[INFO] Training model.")
    x_tr, y_tr = dm.get_x_y('train_cf')
    x_tr = x_tr.cpu().numpy()
    y_tr = y_tr.cpu().numpy().reshape(-1)
    print(x_tr.shape)

    x_val, y_val = dm.get_x_y('validate')
    x_val = x_val.cpu().numpy()
    y_val = y_val.cpu().numpy().reshape(-1)

    probe = LogisticProbe(seed=config["seed"])
    probe.fit(x_tr, y_tr)

    print("[INFO] Evaluating train set performance.")
    train_config["tr_acc"] = probe.score(x_tr, y_tr)
    print(f"[INFO] Train accuracy: {train_config['tr_acc']:.2%}")

    print("[INFO] Evaluating validation set performance.")
    train_config["val_acc"] = probe.score(x_val, y_val)
    print(f"[INFO] Validation accuracy: {train_config['val_acc']:.2%}")

    x_test, y_test = dm.get_x_y('test')
    x_test = x_test.cpu().numpy()
    y_test = y_test.cpu().numpy().reshape(-1)

    print("[INFO] Evaluating test set performance.")
    train_config["test_acc"] = probe.score(x_test, y_test)
    print(f"[INFO] Test accuracy: {train_config['test_acc']:.2%}")

    return probe, train_config


# def train_linear_cf(x_tr, y_tr, dm, config):
#     print("[INFO] Training model.")
#     print(x_tr.shape)
#
#     probe = LogisticRegression(random_state=config['seed'])
#     probe.fit(x_tr, y_tr)
#
#     print("[INFO] Evaluating train set performance.")
#     config['tr_acc_cf'] = probe.score(x_tr, y_tr)
#     print(f"[INFO] Train accuracy: {config['tr_acc_cf']:.2%}")
#
#
#     print("[INFO] Evaluating test set performance.")
#     x_test, y_test = dm.get_x_y('test')
#     x_test = x_test.cpu().numpy()
#     y_test = y_test.cpu().numpy().reshape((-1,))
#     config['test_acc_cf'] = probe.score(x_test, y_test)
#     print(f"[INFO] Test accuracy: {config['test_acc_cf']:.2%}")
#
#     return probe, config
#
