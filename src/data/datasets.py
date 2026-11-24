import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """Class which represents an image dataset where its labels are stored in
     a pandas DataFrame. Each row representing a sample. A column containing
     the images filenames."""

    def __init__(self, df, features, transform_fn, target_transform_fn,
                 filename_col, imgs_path, device):
        self.df = df
        self.features = features
        self.transform_fn = transform_fn
        self.target_transform_fn = target_transform_fn
        self.filename_col = filename_col
        self.imgs_path = imgs_path
        self.device = device

    def _load_image(self, img_file):
        """Loads an image and returns it."""
        return Image.open(self.imgs_path / img_file)

    def _get_label(self, index):
        """Loads the index-th sample's labels from the dataframe."""
        return [self.df.at[index, out_class] for out_class in self.features]

    def __getitem__(self, index):
        """Returns the index-th sample - image and label (X, Y)."""
        img = self._load_image(self.df.at[index, self.filename_col])
        target = self._get_label(index)

        # preprocess
        img = self.transform_fn(img)
        target = self.target_transform_fn(target)

        return img, target

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.df)

class GenericDataset(Dataset):
    def __init__(self, df, load_input, transform, out_features, dataset_cfg,
                 target_transform=None):
        self.df = df.reset_index(drop=True)
        self.load_input = load_input
        self.transform = transform
        self.out_features = out_features
        self.dataset_cfg = dataset_cfg
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        raw = self.load_input(self.df, i, self.dataset_cfg)
        x = self.transform(raw)

        y = torch.tensor(self.df.loc[i, self.out_features].values.astype('float32'))
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y