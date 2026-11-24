import sys
import glob
import os
import random

from src.generation.xtrains.config_generator import generate_train_config
from src.generation.xtrains.constants import *
from src.generation.xtrains.utils import *
from src.utils.pickle import numpy_to_python

NUM_OF_SAMPLES = 100_000
RANDOM_SEED = 42

RENDER = False

if __name__ == "__main__":
    # if needed create folders
    os.makedirs(PATH_TO_IMAGES, exist_ok=True)
    # os.makedirs(PATH_TO_IMAGES_NPZ, exist_ok=True)

    # initialize random seeds
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # store the trains' configurations
    configs = []
    configs_dict = {}

    print("[INFO] Loading background images...")
    backgrounds = [f for f in glob.glob(PATH_TO_BACKGROUNDS)]

    for sample_num in range(0, NUM_OF_SAMPLES):
        if sample_num % math.ceil(NUM_OF_SAMPLES * 0.05) == 0:
            print("[INFO] Generating example: {} of {}".format(sample_num, NUM_OF_SAMPLES))

        image_id = f"{sample_num:09d}"
        image_file = f'{image_id}.png'

        config = generate_train_config(image_id, backgrounds)
        config["image_id"] = image_file
        config["render_seed"] = sample_num

        configs.append(numpy_to_python(config))
        configs_dict[image_file] = numpy_to_python(config)

        if RENDER:
            try:
                render(config, config["image_id"])
            except Exception as e:
                print(f"[ERROR] Error in rendering index {sample_num}: {str(e)}", file=sys.stderr)

    print("[INFO] Generating config file...")
    generate_config_file(PATH_TO_CONFIG, configs)

    # print("[INFO] Generating config file as dict...")
    # generate_config_file(PATH_TO_CONFIG_DICT, configs_dict)
    #
    # print("[INFO] Generating CSV file...")
    # generate_csv(PATH_TO_CSV, configs)

    # print("[INFO] Compressing images...")
    # compress_images(PATH_TO_IMAGES, PATH_TO_IMAGES_NPZ)