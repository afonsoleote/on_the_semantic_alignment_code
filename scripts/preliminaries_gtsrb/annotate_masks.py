import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path as MplPath

from configs.dataset import GTSRBConfig

CSV_PATH = GTSRBConfig.labels_path
IMAGES_DIR = GTSRBConfig.images_path
NAME_COL = GTSRBConfig.img_path_col
OUT = 'attribute_parts.json'
MAX_PER_CONCEPT = 200
SAVE_EVERY = 1

CONCEPTS = [ 'Symbol',
             'Black_Bar',
             'Black_Symbol',
             'Symbol_Speed80',
             'Red_Ground',
             'Post',
             'White_Ground'
    ]

df = pd.read_csv(CSV_PATH)
df = df[df['train_data'] == 0].reset_index(drop=True)

boxes = json.load(open(OUT)) if Path(OUT).exists() else {}

print('already annotated per concept:')
for concept in CONCEPTS:
    done = boxes.get(concept, {})
    n_done = len(done)
    total = min(len(df[df[concept] == 1]), MAX_PER_CONCEPT)
    print(f'  {concept:<18} {n_done}/{total}')
print()

tasks = []
for concept in CONCEPTS:
    done = boxes.get(concept, {})
    n_done = len(done)
    target = list(df[df[concept] == 1][NAME_COL])[:MAX_PER_CONCEPT]
    remaining = [n for n in target if n not in done]
    total = len(target)
    for k, name in enumerate(remaining):
        tasks.append((concept, name, n_done + k + 1, total))


state = {'poly': None, 'polys': [], 'i': 0, 'since_save': 0, 'selector': None}


def flush():
    json.dump(boxes, open(OUT, 'w'))


def on_poly(verts):
    state['poly'] = [[round(float(x), 1), round(float(y), 1)] for x, y in verts]


def save(concept, name, polys):
    boxes.setdefault(concept, {})[name] = polys
    state['since_save'] += 1
    if state['since_save'] >= SAVE_EVERY:
        flush()
        state['since_save'] = 0


def render():
    concept, name, pos, total = tasks[state['i']]
    ax.clear()
    img = np.array(Image.open(IMAGES_DIR / name).convert('RGB'))
    ax.imshow(img)
    for poly in state['polys']:
        xs = [p[0] for p in poly] + [poly[0][0]]
        ys = [p[1] for p in poly] + [poly[0][1]]
        ax.plot(xs, ys, color='cyan', linewidth=2)
    n_committed = len(state['polys'])
    ax.set_title(f'{concept}  [{pos}/{total}]   (session {state["i"] + 1}/{len(tasks)})\n'
                 f'{name}   |   polygons on this image: {n_committed}\n'
                 f'a=add polygon  n=save+next  s=skip  q=quit',
                 fontsize=9)
    ax.axis('off')
    state['poly'] = None
    state['selector'] = PolygonSelector(
        ax, onselect=on_poly,
        props=dict(color='lime', linewidth=2),
        handle_props=dict(markerfacecolor='lime', markersize=8),
    )
    fig.canvas.draw_idle()


def show_new(idx):
    state['polys'] = []
    state['i'] = idx
    render()


def commit_current():
    if state['poly'] is not None:
        state['polys'].append(state['poly'])
        state['poly'] = None


def advance():
    if state['i'] + 1 >= len(tasks):
        flush()
        print('done')
        plt.close(fig)
        return
    show_new(state['i'] + 1)


def on_key(event):
    concept, name, pos, total = tasks[state['i']]
    if event.key == 'a': # add another polygon in the same img
        commit_current()
        render()
        return
    if event.key == 'n': # save and advance
        commit_current()
        save(concept, name, state['polys'] if state['polys'] else None)
        advance()
        return
    if event.key == 'z': # skip (not visible or impossible to intrevnee on)
        advance()
        return
    if event.key == 'q':
        flush()
        plt.close(fig)
        return


if tasks:
    fig, ax = plt.subplots(figsize=(9, 9))
    fig.canvas.mpl_connect('key_press_event', on_key)
    show_new(0)
    plt.show()
    flush()
else:
    print('end')


def polygons_to_mask(polys, shape):
    h, w = shape[:2]
    yy, xx = np.mgrid[:h, :w]
    points = np.vstack((xx.ravel(), yy.ravel())).T
    mask = np.zeros(h * w, dtype=bool)
    for poly in polys:
        mask |= MplPath(poly).contains_points(points)
    return mask.reshape(h, w)