import glob
import os
import random
import sys

import seaborn as sns

from scripts.preliminaries_xtrains.train_renderer_attributes import TrainRendererAttributes

from src.generation.xtrains.constants import *
from src.generation.xtrains.utils import *
from src.utils.pickle import numpy_to_python

NUM_OF_SAMPLES = 10
RANDOM_SEED = 42

RENDER = True

# wagons to show in the illustrative image of some of the xtrains attributes
CONFIGS = [
{   'angle': np.float64(7.551995810103694),
    'background': np.str_('/home/afonso/Desktop/thesis_code/on_the_semantic_reliability/assets/datasets/mc_gill_calibrated_colour_image_database/merry_mexico0211.TIF'),
    'base_height': 20,
    'base_width': 24,
    'cars': [

        ('c0',
         {'cargo_edge': 9,
          'cargo_num': 1,
          'cargo_rad': 2,
          'cargo_type': np.str_('none'),
          'cargo_y': np.int64(7),
          'height': 1,
          'img_shape': (20, 24, 3),
          'roof': np.str_('none'),
          'shape': np.str_('box'),
          'wheels': np.int64(2),
          'width': np.float64(1.42)}),



        (   'c1',
                    {   'cargo_edge': 9,
                        'cargo_num': 1,
                        'cargo_rad': 2,
                        'cargo_type': np.str_('none'),
                        'cargo_y': np.int64(7),
                        'height': 1,
                        'img_shape': (20, 24, 3),
                        'roof': np.str_('none'),
                        'shape': np.str_('box'),
                        'wheels': np.int64(2),
                        'width': np.float64(1.0)}),
],
    'half_space': np.int64(1),
    'id': '000000000',
    'image_id': '000000000.png',
    'linker_y': np.int64(13),
    'render_seed': 0,
    'roof_height': 3,
    'wheel_base': np.int64(13),
    'wheel_radius': np.int64(3)},

{   'angle': np.float64(7.551995810103694),
    'background': np.str_('/home/afonso/Desktop/thesis_code/on_the_semantic_reliability/assets/datasets/mc_gill_calibrated_colour_image_database/merry_mexico0211.TIF'),
    'base_height': 20,
    'base_width': 24,
    'cars': [

        ('c0',
         {'cargo_edge': 9,
          'cargo_num': 1,
          'cargo_rad': 2,
          'cargo_type': np.str_('none'),
          'cargo_y': np.int64(7),
          'height': 1,
          'img_shape': (20, 24, 3),
          'roof': np.str_('none'),
          'shape': np.str_('box'),
          'wheels': np.int64(2),
          'width': np.float64(1.42)}),


        ('c1',
         {'cargo_edge': 9,
          'cargo_num': 1,
          'cargo_rad': 2,
          'cargo_type': np.str_('none'),
          'cargo_y': np.int64(7),
          'height': 1,
          'img_shape': (20, 24, 3),
          'roof': np.str_('none'),
          'shape': np.str_('ellipse'),
          'wheels': np.int64(2),
          'width': np.float64(1.42)}),

        ('c2',
         {'cargo_edge': 9,
          'cargo_num': 1,
          'cargo_rad': 2,
          'cargo_type': np.str_('none'),
          'cargo_y': np.int64(7),
          'height': 1,
          'img_shape': (20, 24, 3),
          'roof': np.str_('none'),
          'shape': np.str_('trapezoid'),
          'wheels': np.int64(2),
          'width': np.float64(1.42)}),

        ('c3',
         {'cargo_edge': 9,
          'cargo_num': 1,
          'cargo_rad': 2,
          'cargo_type': np.str_('none'),
          'cargo_y': np.int64(7),
          'height': 1,
          'img_shape': (20, 24, 3),
          'roof': np.str_('none'),
          'shape': np.str_('superellipse'),
          'wheels': np.int64(2),
          'width': np.float64(1.42)}),

        ('c4',
         {'cargo_edge': 9,
          'cargo_num': 1,
          'cargo_rad': 2,
          'cargo_type': np.str_('none'),
          'cargo_y': np.int64(7),
          'height': 1,
          'img_shape': (20, 24, 3),
          'roof': np.str_('none'),
          'shape': np.str_('hexagon'),
          'wheels': np.int64(2),
          'width': np.float64(1.42)}),

        ('c5',
         {'cargo_edge': 9,
          'cargo_num': 1,
          'cargo_rad': 2,
          'cargo_type': np.str_('none'),
          'cargo_y': np.int64(7),
          'height': 1,
          'img_shape': (20, 24, 3),
          'roof': np.str_('none'),
          'shape': np.str_('reinforced'),
          'wheels': np.int64(2),
          'width': np.float64(1.42)}),



],
    'half_space': np.int64(1),
    'id': '000000000',
    'image_id': '000000000.png',
    'linker_y': np.int64(13),
    'render_seed': 0,
    'roof_height': 3,
    'wheel_base': np.int64(13),
    'wheel_radius': np.int64(3)},

{   'angle': np.float64(7.551995810103694),
    'background': np.str_('/home/afonso/Desktop/thesis_code/on_the_semantic_reliability/assets/datasets/mc_gill_calibrated_colour_image_database/merry_mexico0211.TIF'),
    'base_height': 20,
    'base_width': 24,
    'cars': [

        ('c0',
         {'cargo_edge': 9,
          'cargo_num': 1,
          'cargo_rad': 2,
          'cargo_type': np.str_('none'),
          'cargo_y': np.int64(7),
          'height': 1,
          'img_shape': (20, 24, 3),
          'roof': np.str_('open'),
          'shape': np.str_('box'),
          'wheels': np.int64(2),
          'width': np.float64(1.42)}),


        ('c1',
         {'cargo_edge': 9,
          'cargo_num': 1,
          'cargo_rad': 2,
          'cargo_type': np.str_('none'),
          'cargo_y': np.int64(7),
          'height': 1,
          'img_shape': (20, 24, 3),
          'roof': np.str_('flat'),
          'shape': np.str_('box'),
          'wheels': np.int64(2),
          'width': np.float64(1.42)}),

        ('c2',
         {'cargo_edge': 9,
          'cargo_num': 1,
          'cargo_rad': 2,
          'cargo_type': np.str_('none'),
          'cargo_y': np.int64(7),
          'height': 1,
          'img_shape': (20, 24, 3),
          'roof': np.str_('triangular'),
          'shape': np.str_('box'),
          'wheels': np.int64(2),
          'width': np.float64(1.42)}),

        ('c3',
         {'cargo_edge': 9,
          'cargo_num': 1,
          'cargo_rad': 2,
          'cargo_type': np.str_('none'),
          'cargo_y': np.int64(7),
          'height': 1,
          'img_shape': (20, 24, 3),
          'roof': np.str_('saw'),
          'shape': np.str_('box'),
          'wheels': np.int64(2),
          'width': np.float64(1.42)}),

        ('c4',
         {'cargo_edge': 9,
          'cargo_num': 1,
          'cargo_rad': 2,
          'cargo_type': np.str_('none'),
          'cargo_y': np.int64(7),
          'height': 1,
          'img_shape': (20, 24, 3),
          'roof': np.str_('none'),
          'shape': np.str_('ellipse'),
          'wheels': np.int64(2),
          'width': np.float64(1.42)}),



],
    'half_space': np.int64(1),
    'id': '000000000',
    'image_id': '000000000.png',
    'linker_y': np.int64(13),
    'render_seed': 0,
    'roof_height': 3,
    'wheel_base': np.int64(13),
    'wheel_radius': np.int64(3)},

{   'angle': np.float64(7.551995810103694),
    'background': np.str_('/home/afonso/Desktop/thesis_code/on_the_semantic_reliability/assets/datasets/mc_gill_calibrated_colour_image_database/merry_mexico0211.TIF'),
    'base_height': 20,
    'base_width': 24,
    'cars': [

        ('c0',
         {'cargo_edge': 9,
          'cargo_num': 1,
          'cargo_rad': 3,
          'cargo_type': np.str_('none'),
          'cargo_y': np.int64(7),
          'height': 1,
          'img_shape': (20, 24, 3),
          'roof': np.str_('open'),
          'shape': np.str_('box'),
          'wheels': np.int64(2),
          'width': np.float64(1.42)}),


        ('c1',
         {'cargo_edge': 9,
          'cargo_num': 1,
          'cargo_rad': 3,
          'cargo_type': np.str_('ssquare'),
          'cargo_y': np.int64(7),
          'height': 1,
          'img_shape': (20, 24, 3),
          'roof': np.str_('open'),
          'shape': np.str_('box'),
          'wheels': np.int64(2),
          'width': np.float64(1.42)}),

        ('c2',
         {'cargo_edge': 9,
          'cargo_num': 1,
          'cargo_rad': 2,
          'cargo_type': np.str_('msquare'),
          'cargo_y': np.int64(7),
          'height': 1,
          'img_shape': (20, 24, 3),
          'roof': np.str_('open'),
          'shape': np.str_('box'),
          'wheels': np.int64(2),
          'width': np.float64(1.42)}),

        ('c3',
         {'cargo_edge': 9,
          'cargo_num': 1,
          'cargo_rad': 2,
          'cargo_type': np.str_('xlsquare'),
          'cargo_y': np.int64(7),
          'height': 1,
          'img_shape': (20, 24, 3),
          'roof': np.str_('open'),
          'shape': np.str_('box'),
          'wheels': np.int64(2),
          'width': np.float64(1.42)}),

        ('c4',
         {'cargo_edge': 9,
          'cargo_num': 1,
          'cargo_rad': 3,
          'cargo_type': np.str_('circle'),
          'cargo_y': np.int64(7),
          'height': 1,
          'img_shape': (20, 24, 3),
          'roof': np.str_('open'),
          'shape': np.str_('box'),
          'wheels': np.int64(2),
          'width': np.float64(1.42)}),

        ('c5',
         {'cargo_edge': 9,
          'cargo_num': 1,
          'cargo_rad': 3,
          'cargo_type': np.str_('triangle'),
          'cargo_y': np.int64(7),
          'height': 1,
          'img_shape': (20, 24, 3),
          'roof': np.str_('open'),
          'shape': np.str_('box'),
          'wheels': np.int64(2),
          'width': np.float64(1.42)}),

        ('c6',
         {'cargo_edge': 9,
          'cargo_num': 1,
          'cargo_rad': 3,
          'cargo_type': np.str_('invtriangle'),
          'cargo_y': np.int64(7),
          'height': 1,
          'img_shape': (20, 24, 3),
          'roof': np.str_('open'),
          'shape': np.str_('box'),
          'wheels': np.int64(2),
          'width': np.float64(1.42)}),

        ('c7',
         {'cargo_edge': 9,
          'cargo_num': 1,
          'cargo_rad': 3,
          'cargo_type': np.str_('diamond'),
          'cargo_y': np.int64(7),
          'height': 1,
          'img_shape': (20, 24, 3),
          'roof': np.str_('open'),
          'shape': np.str_('box'),
          'wheels': np.int64(2),
          'width': np.float64(1.42)}),

        ('c8',
         {'cargo_edge': 9,
          'cargo_num': 1,
          'cargo_rad': 3,
          'cargo_type': np.str_('hexagon'),
          'cargo_y': np.int64(7),
          'height': 1,
          'img_shape': (20, 24, 3),
          'roof': np.str_('open'),
          'shape': np.str_('box'),
          'wheels': np.int64(2),
          'width': np.float64(1.42)}),

    ],
    'half_space': np.int64(1),
    'id': '000000000',
    'image_id': '000000000.png',
    'linker_y': np.int64(13),
    'render_seed': 0,
    'roof_height': 3,
    'wheel_base': np.int64(13),
    'wheel_radius': np.int64(3)},
]

label_names = [
    "long\_wagon",
    "short\_wagon",
    "box\_car",
    "ellipse\_car",
    "trapezoid\_car",
    "superellipse\_car",
    "hexagon\_car",
    "reinforced\_car",
    "open\_roof",
    "flat\_roof",
    "triangular\_roof",
    "saw\_roof",
    "ellipse\_roof",
    "empty\_cargo",
    "ssquare\_cargo",
    "msquare\_cargo",
    "xlsquare\_cargo",
    "circle\_cargo",
    "triangle\_cargo",
    "invtriangle\_cargo",
    "diamond\_cargo",
    "hexagon\_cargo",
]
NUM_OF_SAMPLES = len(CONFIGS)

def styling():
    sns.set_theme(context='paper', style='ticks', palette='deep',
                  font_scale=1, color_codes=True, rc=None)
    # plt.rcParams['text.latex.preamble'] = "\\usepackage{times}\n\\usepackage{courier}"
    # plt.rcParams['text.latex.preamble'] = "\\usepackage[T1]{fontenc}\n\\fontfamily{georgia}\\selectfont"
    #plt.rcParams['font.family'] = 'Georgia'
    plt.rcParams['text.latex.preamble'] = "\\usepackage{amsmath}\\newcommand{\indep}{\mathrel{\perp\!\!\!\perp}}"

    params = {'text.usetex': True,
              'font.size': 11,
              'axes.linewidth': 0.5,
              'axes.labelsize': 11,
              'xtick.labelsize': 11,
              'ytick.labelsize': 11,
              'xtick.major.width': 0.2,
              'xtick.major.size': 3,
              'ytick.major.width': 0.2,
              'ytick.major.size': 3,
              'lines.linewidth': 1,
              }
    plt.rcParams.update(params)

    params = {'text.usetex': True,
              'font.family': 'serif',
              'axes.linewidth': 0.5,}
    plt.rcParams.update(params)


if __name__ == "__main__":
    os.makedirs(PATH_TO_IMAGES, exist_ok=True)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    configs = []
    configs_dict = {}
    labels = []

    print("[INFO] Loading background images...")
    backgrounds = [f for f in glob.glob(PATH_TO_BACKGROUNDS)]

    IMG_HEI = 230 # limit
    IMG_WID = 600 # 515 # limit
    train_img = np.ones((IMG_HEI, IMG_WID, 3))

    dx = 70
    dy = 50
    top = 0
    bottom = 20

    left = 0
    right = 40

    fig, ax = plt.subplots(figsize=(12, 6))

    top = 0
    left = 0

    labels = []
    i = 0
    for sample_num in range(0, NUM_OF_SAMPLES):
        if sample_num % math.ceil(NUM_OF_SAMPLES * 0.05) == 0:
            print("[INFO] Generating example: {} of {}".format(sample_num, NUM_OF_SAMPLES))

        image_id = f"{sample_num:09d}"
        image_file = f'{image_id}.png'

        config = CONFIGS[sample_num]
        config["image_id"] = image_file
        config["render_seed"] = sample_num

        configs.append(numpy_to_python(config))
        configs_dict[image_file] = numpy_to_python(config)
        import pprint
        pp = pprint.PrettyPrinter(indent=4)

        if RENDER:
            try:
                tr = TrainRendererAttributes(config)
                car_imgs = tr.get_car_imgs()
                styling()

                for car_idx, car_img in enumerate(car_imgs):

                    print(car_img.shape)

                    height, width = car_img.shape[:2]

                    ax.imshow(
                        car_img,
                        extent=[
                            left,
                            left + width,
                            top + height,
                            top
                        ],
                        interpolation="none"
                    )


                    ax.text(
                        left + width / 2,
                        top + height + 5,
                        label_names[i],
                        ha="center",
                        va="top",
                        fontsize=10,
                        color="black"
                    )
                    i+=1

                    if config["cars"][1][1]["width"] == 1:
                        left += 75
                    else:
                        left += dx

                    right += dx

            except Exception as e:
                print(
                    f"[ERROR] Error in rendering index {sample_num}: {str(e)}",
                    file=sys.stderr
                )

            top += dy
            bottom += dy
            left = 0
            right = 40

    ax.set_xlim(0, IMG_WID)
    ax.set_ylim(IMG_HEI, 0)

    ax.axis("off")

    plt.show()

