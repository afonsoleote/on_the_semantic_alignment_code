import json

import numpy as np
from PIL import Image

from configs.dataset import GTSRBConfig
from configs.experimental_setup import GTSRBSetup
from src.framework.realization_cf import Realization
from src.generation.utils.bounding_box_based import BoundingBoxInterventionPrimitives as P


class GTSRBRealization(Realization):

    EDITS = {
        'Symbol': P.inpaint_polygon,
        'Black_Bar': P.inpaint_polygon,
        'Black_Symbol': P.inpaint_polygon,
        'Blue': P.invert_polygon,
        'Red_Ground': P.invert_polygon,
        'Symbol_Speed80': P.inpaint_polygon,
        'Post': P.inpaint_polygon,
        'Circular_Shape': P.inpaint_polygon,
        'Triangular_Shape': P.inpaint_polygon,
        'White_Ground': P.invert_polygon,
    }

    def __init__(self, df, images_path, parts_path, setup=GTSRBSetup, name_col=GTSRBConfig.img_path_col):
        self.df = df
        self.images_path = images_path
        self.polygons = json.load(open(parts_path))
        self.name_col = name_col
        self.attributes = setup.attributes

    def image(self, x):
        return np.array(Image.open(self.images_path / self.df.loc[x, self.name_col]).convert('RGB'))

    def realize(self, x, V, Vcf):
        name = self.df.loc[x, self.name_col]
        img = self.image(x)
        for a, value in self.changes(V, Vcf):
            polygons = self.polygons.get(a, {}).get(name)
            if value or not polygons:
                return None
            for polygon in polygons:
                img = self.EDITS[a](img, polygon)
        return img


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd

    from src.framework.counterfactual_intervention import counterfactual_intervention, labeling
    from src.utils.common_paths import GTSRB_IMAGE_PATH

    ATTRIBUTE = 'Symbol'

    df = pd.read_csv(GTSRBConfig.labels_path)
    exp_setup = GTSRBSetup
    K = exp_setup.knowledge(ATTRIBUTE)
    R = GTSRBRealization(df, GTSRB_IMAGE_PATH, GTSRBConfig.bb_parts_path)

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