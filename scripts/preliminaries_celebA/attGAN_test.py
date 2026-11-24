import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image

from src.utils.common_paths import ASSETS_PATH
from src.utils.models.AttGAN import Generator

CKPT      = ASSETS_PATH / "generators" / "attgan.pth"
IMG_PATH  = './imgs/img2.png'
TEST_INT  = 1.0
THRES_INT = 0.5
DEV       = 'cpu'

ATTRS = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows',
         'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard',
         'Pale_Skin', 'Young']

#        Bald Bangs Blk Bln Brn Bushy Glass Male MouthO Must NoBeard Pale Young
SRC =   [ 0,   0,    0,  0,  1,   0,    0,    1,   1,     0,   1,      0,   1 ]

G = Generator(enc_dim=64, enc_layers=5, enc_norm_fn='batchnorm', enc_acti_fn='lrelu',
              dec_dim=64, dec_layers=5, dec_norm_fn='batchnorm', dec_acti_fn='relu',
              n_attrs=len(ATTRS), shortcut_layers=1, inject_layers=1, img_size=128)
G.load_state_dict(torch.load(CKPT, map_location='cpu')['G'])
G.to(DEV).eval()

tf = T.Compose([
    T.Resize(191),
    T.CenterCrop(170), T.Resize(128), T.ToTensor(),
    T.Normalize((0.5,)*3, (0.5,)*3),
])
x = tf(Image.open(IMG_PATH).convert('RGB')).unsqueeze(0).to(DEV)

def target_label(attr):
    att = torch.tensor([SRC], dtype=torch.float, device=DEV)
    i = ATTRS.index(attr)
    att[0, i] = 1.0 - att[0, i]

    if att[0, i] != 0:
        for g in [['Black_Hair', 'Blond_Hair', 'Brown_Hair'], ['Mustache', 'No_Beard']]:
            if attr in g:
                for n in g:
                    if n != attr:
                        att[0, ATTRS.index(n)] = 0.0
        if attr == 'Bald':  att[0, ATTRS.index('Bangs')] = 0.0
        if attr == 'Bangs': att[0, ATTRS.index('Bald')]  = 0.0

    att_ = (att * 2 - 1) * THRES_INT
    att_[0, i] = att_[0, i] * TEST_INT / THRES_INT
    return att_

labels = torch.cat([target_label(a) for a in ATTRS], dim=0)
imgs   = x.repeat(len(ATTRS), 1, 1, 1)
with torch.no_grad():
    cfs = G(imgs, labels)

def to_img(t):
    return ((t + 1) / 2).clamp(0, 1).permute(1, 2, 0).cpu().numpy()

cols = 7
rows = -(-(len(ATTRS) + 1) // cols)
fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2.2 * rows))
axes = axes.ravel()

axes[0].imshow(to_img(x[0])); axes[0].set_title('original', fontsize=9)
for k, attr in enumerate(ATTRS):
    on = SRC[ATTRS.index(attr)] == 0
    axes[k + 1].imshow(to_img(cfs[k]))
    axes[k + 1].set_title(f"{'+' if on else '−'}{attr}", fontsize=8)
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()