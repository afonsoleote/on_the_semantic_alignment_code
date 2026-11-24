import copy

import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import DataLoader

from src.data.dataframe_data_handling import default_transform, default_target_transform, balance_binary_feature, \
    split_indices
from src.data.datasets import ImageDataset
from src.generation.xtrains.utils import render_in_mem
from src.probing.activations_extractor import ActivationsExtractor
from src.utils.nn_utils.checkpoint_utils import load_model


class ImageDataModuleXTRAINS:
    def __init__(self, out_features, id_col, batch_size,
                 csv_path, imgs_path, device, seed,
                 other_features=None,
                 dtype_dict=None,
                 transforms=None,
                 target_transform=None,
                 collate_fn=None,
                 num_workers=0):
        super().__init__()

        self.out_features = out_features
        self.id_col = id_col
        self.transforms = transforms or default_transform()
        self.target_transform = target_transform or default_target_transform
        self.batch_size = batch_size
        self.csv_path = csv_path
        self.imgs_path = imgs_path
        self.device = device

        self.seed = seed if seed is not None else 42

        self.other_features = other_features or []
        self.collate_fn = collate_fn
        self.num_workers = num_workers

        self.cols = ([self.id_col] + self.out_features + self.other_features)
        self.dtype_dict = dtype_dict or {out_class: 'float32'
                                         for out_class in self.out_features}

        self.df = pd.read_csv(self.csv_path,
                              # usecols=self.cols,
                              dtype=self.dtype_dict)

        # data splits
        self.tr = None
        self.val = None
        self.test = None

        # dataloaders
        self.tr_dl = None
        self.val_dl = None
        self.test_dl = None
        self.all_dl = None


    def set_default_splits(self, tr_size, val_size, test_size, verbose=True):

        binary_feat = self.out_features[0]

        temp_df = balance_binary_feature(
            self.df,
            binary_feat,
            total_samples=tr_size + val_size + test_size,
            seed=self.seed
        )

        self.tr, self.val, self.test = split_indices(
            temp_df,
            tr_size,
            val_size,
            test_size,
            stratify=binary_feat,
            seed=self.seed
        )

        self.renew_dls()

        if verbose:
            for split_name, split_idx in zip(
                    ["Train", "Validation", "Test"],
                    [self.tr, self.val, self.test]
            ):
                counts = temp_df.loc[split_idx, binary_feat].value_counts()
                pos = counts.get(1, 0)
                neg = counts.get(0, 0)
                print(f"{split_name} split: {pos} positive, {neg} negative samples")

    def get_dataset(self, stage, index_split=None):
        """Returns a dataset for the provided split with the given
         specifications."""
        temp_df = self.df if index_split is None else self.df.loc[index_split]

        if isinstance(self.transforms, dict):
            if stage == 'fit':
                transform = self.transforms['fit']
            else:
                transform = self.transforms['predict']
        else:
            transform = self.transforms

        return ImageDataset(temp_df.reset_index(drop=True),
                                    self.out_features,
                                    transform,
                                    self.target_transform,
                                    self.id_col,
                                    self.imgs_path,
                                    self.device)

    def _get_dataloader(self, dataset, shuffle):
        """Returns a dataloader for the provided dataset with the given
         specifications."""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=self.device == 'cuda',
            collate_fn=self.collate_fn)

    def setup(self, stage):
        if stage == 'fit' and self.tr_dl is None:
            self.tr_dl = self._get_dataloader(
                self.get_dataset(stage, index_split=self.tr),
                True)

        if (stage == 'fit' or stage == 'validate') and self.val_dl is None:
            self.val_dl = self._get_dataloader(
                self.get_dataset(stage, index_split=self.val),
                False)

        if stage == 'test' and self.test_dl is None:
            self.test_dl = self._get_dataloader(
                self.get_dataset(stage, index_split=self.test),
                False)

        if stage == 'predict' and self.all_dl is None:
            self.all_dl = self._get_dataloader(
                self.get_dataset(stage),
                False)

    def train_dataloader(self):
        return self.tr_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl

    def train_val_dataloader(self):
        return self._get_dataloader(
            self.get_dataset('fit', index_split=self.tr.append(self.val)),
            True)

    def predict_dataloader(self):
        return self.all_dl

    def renew_dls(self):
        self.tr_dl = None
        self.val_dl = None
        self.test_dl = None
        self.all_dl = None

    @staticmethod
    def sample_exact_correlation(size, _df, target, other, corr, rng):
        assert 0 <= corr <= 1

        n11 = int(size * (1 + corr) / 4)
        n00 = int(size * (1 + corr) / 4)
        n10 = int(size * (1 - corr) / 4)
        n01 = int(size * (1 - corr) / 4)

        print(n11, n00, n10, n01)

        # tanbem se pode ignorar este passo, é so para ter o numero exato de samples pedido com a correlacao pedida
        total = n11 + n00 + n10 + n01
        remainder = size - total
        n11 += remainder

        counts = {
            (1, 1): n11,
            (0, 0): n00,
            (1, 0): n10,
            (0, 1): n01,
        }

        idxs = []

        for (t_val, o_val), n in counts.items():
            subset = _df[(_df[target] == t_val) & (_df[other] == o_val)]

            if len(subset) < n:
                raise ValueError(f"Not enough samples for group {(t_val, o_val)}")

            sampled = subset.sample(n=n, random_state=rng).index
            idxs.append(sampled)

        return pd.Index(np.concatenate(idxs))

    def set_correlated_splits(self, target, other, corr, tr_size, val_size, test_size, verbose=True):
        rng = np.random.RandomState(self.seed)
        df = self.df.copy()

        pool = df
        self.tr = self.sample_exact_correlation(tr_size, pool, target, other, corr, rng)

        pool = pool.drop(self.tr)
        self.val = self.sample_exact_correlation(val_size, pool, target, other, corr, rng)

        pool = pool.drop(self.val)
        self.test = self.sample_exact_correlation(test_size, pool, target, other, corr, rng)

        self.renew_dls()

        if verbose:
            def compute_corr(idx):
                sub = df.loc[idx]
                return sub[target].corr(sub[other])

            print(f"Train correlation: {compute_corr(self.tr):.3f}")
            print(f"Validation correlation: {compute_corr(self.val):.3f}")
            print(f"Test correlation: {compute_corr(self.test):.3f}")

    def get_x_y(self, stage, device=None):
        if stage not in ['train', 'validate', 'test', 'predict', 'trainval', 'train_cf']:
            raise ValueError(f'Invalid stage: {stage}. Please specify one of: '
                             f'train, validate, test, predict.')
        if stage != 'train_cf':
            if stage == 'train' or stage == 'trainval':
                self.setup('fit')
            else:
                self.setup(stage)

            if stage == 'fit' or stage == 'train':
                dl = self.train_dataloader()
            elif stage == 'validate':
                dl = self.val_dataloader()
            elif stage == 'test':
                dl = self.test_dataloader()
            elif stage == 'predict':
                dl = self.predict_dataloader()
            elif stage == 'trainval':
                dl = self.train_val_dataloader()
            # elif stage == 'train_cf':


            if device is not None and device != self.device:
                x, y = zip(*((x.to(device), y.to(device))
                             for (x, y) in dl))
            else:
                x, y = zip(*dl)

            x = torch.cat(x, dim=0).detach()
            y = torch.cat(y, dim=0).detach()

        else: # stage == "linear_cf"
            x = self.cf_train_x
            y = self.cf_train_y

        return x, y

    def remove_nans(self, concept):
        self.df = self.df.dropna(subset=[concept])

    def get_counterfactual_pairs(self, ids, configs, intervention):
        imgs = []
        imgs_cf = []
        for f in ids:
            path = self.imgs_path / f

            # imgs.append(self.transforms(Image.open(path))) # todo
            img = render_in_mem(copy.deepcopy(configs[f]))
            imgs.append(self.transforms(img))

            img_cf = intervention.img_result(copy.deepcopy(configs[f]))
            imgs_cf.append(self.transforms(img_cf))

        return torch.stack(imgs), torch.stack(imgs_cf)

    def get_imgs_from_config(self, ids,  configs):
        imgs = []
        for f in ids:
            path = self.imgs_path / f

            img = Image.open(path) # todo
            # img = render_in_mem(copy.deepcopy(configs[f]))
            imgs.append(self.transforms(img))

        return torch.stack(imgs)


    def set_cf_train(self, x, x_cf):
        self.cf_train_x = torch.cat([x, x_cf], dim=0)
        self.cf_train_y = torch.cat([
            torch.ones(len(x), 1, dtype=torch.float32, device=x.device),
            torch.zeros(len(x_cf), 1, dtype=torch.float32, device=x.device),
        ], dim=0)

    def __len__(self):
        return len(self.df)



    def get_filenames(self, idx, only_positive=False):
        subset = self.df.loc[idx]
        if only_positive:
            return subset.loc[subset[self.out_features[0]] == 1, self.id_col]

        return subset[self.id_col]

    def get_filenames_subset(self, idx, pos=True):
        subset = self.df.loc[idx]
        if pos:
            return subset.loc[subset[self.out_features[0]] == 1, self.id_col]
        else:
            return subset.loc[subset[self.out_features[0]] == 0, self.id_col]

    def get_filenames_condition(self, target, protected, target_value=1, protected_value=1, amount=None):
        if amount is None:
            amount = int(len(self.test) / 2)

        combined_idxs = self.tr.union(self.val)
        df_subset = self.df.loc[~self.df.index.isin(combined_idxs)]

        if target_value is None:
            mask = (df_subset[protected] == protected_value)
        else:
            mask = ((df_subset[target] == target_value) & (df_subset[protected] == protected_value))

        filtered = df_subset.loc[mask]

        sampled = filtered.sample(n=amount, random_state=self.seed)

        return sampled[self.id_col]

    def get_filenames_condition_test(self, target, protected, target_value=1, protected_value=1, amount=None):
        if amount is None:
            amount = int(len(self.test) / 2)

        df_subset = self.df.loc[self.test]

        if target_value is None:
            mask = (df_subset[protected] == protected_value)
        else:
            mask = ((df_subset[target] == target_value) & (df_subset[protected] == protected_value))
        filtered = df_subset.loc[mask]

        sampled = filtered.sample(n=amount, random_state=self.seed)

        return sampled[self.id_col]


    def get_conditioned_filenames(self, protected, conditionals=(),
                                  protected_value=1, amount=None):
        df_subset = self.df.loc[self.test_alg]

        mask = (df_subset[protected] == protected_value)

        if conditionals is not None:
            for val, col in conditionals:
                mask = mask & (df_subset[col] == val)

        filtered = df_subset.loc[mask]

        sampled = filtered.sample(n=amount, random_state=self.seed)

        return sampled[self.id_col]


class ProbingDataModuleXTRAINS(ImageDataModuleXTRAINS):
    def __init__(self, out_features, id_col, batch_size,
                 csv_path, imgs_path, device, seed,
                 model_path,
                 to_extract,
                 other_features=None,
                 dtype_dict=None,
                 transforms=None,
                 target_transform=None,
                 collate_fn=None,
                 num_workers=0):
        super().__init__(
            out_features=out_features,
            id_col=id_col,
            batch_size=batch_size,
            csv_path=csv_path,
            imgs_path=imgs_path,
            device=device,
            seed=seed,
            other_features=other_features,
            dtype_dict=dtype_dict,
            transforms=transforms,
            target_transform=target_transform,
            collate_fn=collate_fn,
            num_workers=num_workers
        )

        self.to_extract = to_extract

        model = load_model(model_path, self.device)
        self.extractor = ActivationsExtractor(model, to_extract)

    def _get_dataloader(self, dataset, shuffle):

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=self.device == 'cuda',
            collate_fn=self.probing_collate
        )

    def probing_collate(self, batch):
        """
        Collate function that extracts activations on the fly.
        Input: batch of (x, y) from dataset
        Output: (activations, y)
        """
        x, y = zip(*batch)  # unzip batch
        x = torch.stack(x).to(self.device)
        y = torch.stack([
            t.clone().detach() if isinstance(t, torch.Tensor) else torch.tensor(t)
            for t in y
        ]).to(self.device)
        acts = self.extractor.extract(x, as_numpy=False, flatten=True)

        return acts, y

    def get_activations_counterfactual_pairs(self, ids, configs, intervention):
        x, x_cf = self.get_counterfactual_pairs(ids, configs, intervention)
        x_acts = self.extractor.extract(x, as_numpy=False, flatten=True)
        x_acts_cf = self.extractor.extract(x_cf, as_numpy=False, flatten=True)
        return x_acts, x_acts_cf

    def get_activations_from_imgs(self, x):
        x_acts = self.extractor.extract(x, as_numpy=False, flatten=True)
        return x_acts



