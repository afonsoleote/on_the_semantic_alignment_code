import random

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.utils.common_paths import DATASETS_PATH

ROOT = DATASETS_PATH / "CUB_200_2011"
IMAGES_DIR = ROOT / 'images'
RADIUS_RATIO = 0.12

PART_NAMES = {
    1: 'back', 2: 'beak', 3: 'belly', 4: 'breast', 5: 'crown', 6: 'forehead',
    7: 'left_eye', 8: 'left_leg', 9: 'left_wing', 10: 'nape', 11: 'right_eye',
    12: 'right_leg', 13: 'right_wing', 14: 'tail', 15: 'throat',
}

RADIUS_RATIO = {
    1: 0.18,  # back    (large, extended)
    2: 0.08,  # beak    (small)
    3: 0.20,  # belly   (large)
    4: 0.10,  # breast  (large)
    5: 0.08,  # crown   (small-medium)
    6: 0.08,  # forehead(small)
    7: 0.02,  # left_eye(tiny)
    8: 0.07,  # left_leg(small)
    9: 0.18,  # left_wing (large, extended)
    10: 0.09,  # nape    (small-medium)
    11: 0.02,  # right_eye (tiny)
    12: 0.07,  # right_leg (small)
    13: 0.18,  # right_wing (large, extended)
    14: 0.18,  # tail    (large, extended)
    15: 0.05,  # throat  (medium)
}


def read_id_to_file():
    d = {}
    with open(ROOT / 'images.txt') as f:
        for line in f:
            img_id, path = line.split()
            d[int(img_id)] = path
    return d


def read_locs():
    d = {}
    with open(ROOT / 'parts' / 'part_locs.txt') as f:
        for line in f:
            img_id, part_id, x, y, vis = line.split()
            d.setdefault(int(img_id), {})[int(part_id)] = (float(x), float(y), int(vis))
    return d


def read_bboxes():
    d = {}
    with open(ROOT / 'bounding_boxes.txt') as f:
        for line in f:
            img_id, x, y, w, h = line.split()
            d[int(img_id)] = tuple(map(float, (x, y, w, h)))
    return d


id_to_file = read_id_to_file()
locs = read_locs()
bboxes = read_bboxes()

img_id = random.choice(list(id_to_file.keys()))
img = np.array(Image.open(IMAGES_DIR / id_to_file[img_id]).convert('RGB'))
bx, by, bw, bh = bboxes[img_id]
bird_size = max(bw, bh)

print(f'image_id = {img_id}')
print(f'file     = {id_to_file[img_id]}')
print(f'bbox     = ({bx:.0f}, {by:.0f}, {bw:.0f}, {bh:.0f})  bird_size = {bird_size:.0f}px')

parts = sorted(PART_NAMES.keys())
cols = 5
rows = -(-len(parts) // cols)
fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
axes = axes.ravel()

for ax, part_id in zip(axes, parts):
    ax.imshow(img)
    x, y, vis = locs[img_id].get(part_id, (0, 0, 0))
    radius = int(RADIUS_RATIO[part_id] * bird_size)
    if vis:
        ax.add_patch(patches.Rectangle((x - radius, y - radius), 2 * radius, 2 * radius,
                                       fill=False, edgecolor='red', linewidth=1.5))
        ax.plot(x, y, 'r.', markersize=8)
        ax.set_title(f'{part_id} {PART_NAMES[part_id]}  r={radius}px '
                     f'({RADIUS_RATIO[part_id]:.0%})', fontsize=8)
    else:
        ax.set_title(f'{part_id} {PART_NAMES[part_id]} (not visible)', fontsize=8, color='gray')
    ax.axis('off')

for ax in axes[len(parts):]:
    ax.axis('off')

fig.suptitle(f'image {img_id}  |  bird_size = {bird_size:.0f}px', fontsize=12)
plt.tight_layout()
plt.show()
