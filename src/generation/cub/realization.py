import numpy as np
import pandas as pd
from PIL import Image

from configs.dataset import CUBConfig
from configs.experimental_setup import CUBSetup
from src.framework.realization_cf import Realization
from src.generation.utils.bounding_box_based import BoundingBoxInterventionPrimitives as P

# attribute bounding box id
BEAK, BELLY, BREAST, CROWN, FOREHEAD, L_EYE, NAPE, R_EYE, THROAT = 2, 3, 4, 5, 6, 7, 10, 11, 15

# circle radius of each body part
RADIUS = {BEAK: 0.15, CROWN: 0.09, FOREHEAD: 0.07, NAPE: 0.08, BREAST: 0.16,
          BELLY: 0.16, THROAT: 0.11, L_EYE: 0.017, R_EYE: 0.017}

# blur level
BLUR = 0.20

# attribute: (body parts, image edit)
REGIONS = {
    'attr_4': ([BEAK], P.blur_circle),
    'attr_5': ([BEAK], P.blur_circle),
    'attr_104': ([CROWN, FOREHEAD, NAPE, THROAT], P.blur_circle),
    'attr_112': ([BREAST], P.invert_circle),
    'attr_134': ([THROAT], P.invert_circle),
    'attr_148': ([L_EYE, R_EYE], P.invert_circle),
    'attr_198': ([BELLY], P.invert_circle),
    'attr_204': ([BELLY], P.invert_circle),
    'attr_306': ([CROWN], P.invert_circle),
    'attr_307': ([CROWN], P.invert_circle),
}


class CUBRealization(Realization):
    def __init__(self, df, images_path, ids_path, parts_path, boxes_path, setup=CUBSetup,
                 name_col=CUBConfig.img_path_col):
        self.df = df
        self.images_path = images_path
        self.name_col = name_col
        self.attributes = setup.attributes

        ids = pd.read_csv(ids_path)
        col = [c for c in ids.columns if c != 'image_id'][0]
        self.ids = dict(zip(ids[col], ids['image_id']))

        self.parts = {}
        for line in open(parts_path):
            i, part, px, py, visible = line.split()
            self.parts.setdefault(int(i), {})[int(part)] = (float(px), float(py), int(visible))

        self.boxes = {}
        for line in open(boxes_path):
            i, bx, by, bw, bh = line.split()
            self.boxes[int(i)] = (float(bx), float(by), float(bw), float(bh))

    def image(self, x):
        return np.array(Image.open(self.images_path / self.df.loc[x, self.name_col]).convert('RGB'))

    def part_box(self, i, part, shape):
        px, py, visible = self.parts[i].get(part, (0, 0, 0))
        if not visible:
            return None
        _, _, bw, bh = self.boxes[i]
        r = int(RADIUS[part] * max(bw, bh))
        h, w = shape[:2]
        return max(int(px) - r, 0), max(int(py) - r, 0), min(int(px) + r, w), min(int(py) + r, h)

    def edit(self, img, i, a):
        parts, primitive = REGIONS[a]
        for part in parts:
            box = self.part_box(i, part, img.shape)
            if box is None:
                continue
            if primitive is P.blur_circle:
                img = primitive(img, box, BLUR * max(box[2] - box[0], box[3] - box[1]))
            else:
                img = primitive(img, box)
        return img

    def realize(self, x, V, Vcf):
        i = int(self.ids[self.df.loc[x, self.name_col]])
        img = self.image(x)
        for a, value in self.changes(V, Vcf):
            if value:
                return None
            edited = self.edit(img, i, a)
            if np.array_equal(edited, img):
                return None
            img = edited
        return img


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from src.framework.counterfactual_intervention import counterfactual_intervention, labeling
    from src.utils.common_paths import CUB_BB_PATH, CUB_CSV_PATH, CUB_IMAGE_PATH, CUB_PARTS_PATH

    ATTRIBUTE = 'attr_204'

    df = pd.read_csv(CUB_CSV_PATH)
    exp_setup = CUBSetup
    K = exp_setup.knowledge(ATTRIBUTE)
    R = CUBRealization(df, CUB_IMAGE_PATH, CUB_CSV_PATH, CUB_PARTS_PATH, CUB_BB_PATH)
    for x in df.index[df[ATTRIBUTE] == 1]:
        V = exp_setup.extend(labeling(df, x, exp_setup.attributes), ATTRIBUTE)
        x_cf = counterfactual_intervention(x, V, ATTRIBUTE, K, R)
        if x_cf is None:
            continue
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(R.image(x))
        ax[0].set_title('before')
        ax[1].imshow(x_cf)
        ax[1].set_title('after')
        for axis in ax:
            axis.axis('off')
        fig.suptitle(f'{ATTRIBUTE}, {df.loc[x, R.name_col]}')
        plt.show()