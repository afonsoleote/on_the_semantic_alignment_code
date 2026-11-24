import torch
import torchvision.transforms as T
from PIL import Image

from configs.dataset import CelebaConfig
from configs.experimental_setup import CelebASetup
from src.framework.realization_cf import Realization
from src.utils.models.AttGAN import Generator


class CelebaRealization(Realization):
    def __init__(self, df, images_path, ckpt_path, setup=CelebASetup, name_col=CelebaConfig.img_path_col,
                 device='cpu', thres_int=0.5, test_int=1.0, img_size=128, crop=170, resize_pre=190):
        self.df = df
        self.images_path = images_path
        self.name_col = name_col
        self.device = device
        self.thres_int = thres_int
        self.test_int = test_int
        self.attributes = setup.attributes
        self.G = Generator(n_attrs=len(self.attributes), shortcut_layers=1, inject_layers=1,
                           img_size=img_size)
        self.G.load_state_dict(torch.load(ckpt_path, map_location='cpu')['G'])
        self.G.to(device).eval()
        self.tf = T.Compose([T.Resize(resize_pre), T.CenterCrop(crop), T.Resize(img_size),
                             T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])

    def image(self, x):
        img = self.tf(Image.open(self.images_path / self.df.loc[x, self.name_col]).convert('RGB'))
        return ((img + 1) / 2).clamp(0, 1).permute(1, 2, 0).numpy()

    def realize(self, x, V, Vcf):
        img = Image.open(self.images_path / self.df.loc[x, self.name_col]).convert('RGB')
        img = self.tf(img).unsqueeze(0).to(self.device)
        att = torch.tensor([[float(Vcf[a]) for a in self.attributes]])
        att = (att * 2 - 1) * self.thres_int
        for a, _ in self.changes(V, Vcf):
            j = self.attributes.index(a)
            att[0, j] = att[0, j] * self.test_int / self.thres_int
        with torch.no_grad():
            out = self.G(img, att.to(self.device))[0].cpu()
        return ((out + 1) / 2).clamp(0, 1).permute(1, 2, 0).numpy()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd

    from src.framework.counterfactual_intervention import counterfactual_intervention, labeling
    from src.utils.common_paths import ASSETS_PATH, CELEBA_CSV_PATH, CELEBA_IMAGE_PATH  # adjust

    ATTRIBUTE = 'Blond_Hair'

    df = pd.read_csv(CELEBA_CSV_PATH)
    exp_setup = CelebASetup
    K = exp_setup.knowledge(ATTRIBUTE)
    R = CelebaRealization(df, CELEBA_IMAGE_PATH, ASSETS_PATH / 'generators' / 'attgan.pth', crop=190)

    for x in df.index[df[ATTRIBUTE] == 0][:10]:
        V = exp_setup.extend(labeling(df, x, exp_setup.attributes), ATTRIBUTE)
        x_cf = counterfactual_intervention(x, V, ATTRIBUTE, K, R)
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(R.image(x))
        ax[0].set_title('before')
        ax[1].imshow(x_cf)
        ax[1].set_title(f'after: {ATTRIBUTE}')
        for axis in ax:
            axis.axis('off')
        plt.show()