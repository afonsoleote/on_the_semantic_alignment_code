import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms

BALANCE_ERR = "Got total_samples {}, but only {} {} samples are available."
POSITIVE = 'positive'
NEGATIVE = 'negative'

def default_transform():
    """Default transform for image data."""
    return transforms.ToTensor()


def default_target_transform(target):
    """Default target transform for label data."""
    # pin_memory instead device=self.device
    return torch.as_tensor(target, dtype=torch.float32)  # device=self.device

def shuffle_df(df, seed=None):
    """Return a shuffled dataframe."""
    return df.sample(frac=1, random_state=seed)


def balance_binary_feature(df, balance_feat, total_samples=None, seed=None):
    """Balances the data with respect to the given 'balance_feat'.
    Assumes binary values (0/1)."""
    df = shuffle_df(df, seed)
    if total_samples is None:
        pos_samples = int(df[balance_feat].sum())
        pos_samples = neg_samples = min(pos_samples, len(df) - pos_samples)
    else:
        pos_samples = total_samples // 2
        neg_samples = total_samples - total_samples // 2

    mask = df[balance_feat].values == 1
    pos = df[mask][:pos_samples]
    neg = df[~mask][:neg_samples]

    if len(pos) < pos_samples:
        raise ValueError(BALANCE_ERR.format(total_samples, POSITIVE, len(pos)))
    if len(neg) < neg_samples:
        raise ValueError(BALANCE_ERR.format(total_samples, NEGATIVE, len(neg)))

    return shuffle_df(pd.concat([pos, neg]), seed)


def split_indices(df, tr_size, val_size, test_size, stratify=None, seed=None):
    remaining_idx = df.index

    # train split
    tr_idx, remaining_idx = train_test_split(
        remaining_idx,
        train_size=tr_size,
        random_state=seed,
        stratify=df.loc[remaining_idx, stratify] if stratify else None
    )

    # validation / test split
    val_idx, test_idx = train_test_split(
        remaining_idx,
        train_size=val_size,
        test_size=test_size,
        random_state=seed,
        stratify=df.loc[remaining_idx, stratify] if stratify else None
    )

    return tr_idx, val_idx, test_idx

def load_image_input(df, i, cfg):
    return Image.open(cfg.images_path / df[cfg.img_path_col].iloc[i])

def load_image_input_cub(df, i, cfg):
    return Image.open(cfg.images_path / df[cfg.img_path_col].iloc[i]).convert('RGB')

def load_row_input(df, i, cfg):
    return df.iloc[[i]]

def balanced_sample(dm, pool_idx, size):
    rng = np.random.default_rng(dm.seed)
    sub = dm.df.loc[pool_idx]
    groups = sub.groupby(list(dm.out_features)).groups
    members = [np.array(g) for g in groups.values()]

    per = size // len(members)
    per = min(per, min(len(m) for m in members))

    picked = []
    for m in members:
        picked.extend(rng.choice(m, size=per, replace=False))

    if per * len(members) < size:
        print(f'[WARN] requested {size}, smallest class has '
              f'{min(len(m) for m in members)}, returning {per * len(members)} pos/neg')

    return np.array(picked)