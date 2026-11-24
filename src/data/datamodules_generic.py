import copy

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.dataframe_data_handling import balanced_sample
from src.data.datasets import GenericDataset
from src.generation.adult.realization import AdultInterventionRegistryTabular
from src.generation.celeba.realization import AttGANModel
from src.generation.cub.realization import CubInterventionRegistry
from src.generation.gtsrb.realization import GtsrbInterventionRegistry
from src.generation.xtrains.intervention_registry import InterventionManagerXTRAINS
from src.generation.xtrains.utils import filter_configs, render_in_mem
from src.probing.activations_extractor import ActivationsExtractor
from src.utils.common_paths import DATASETS_PATH
from src.utils.nn_utils.checkpoint_utils import load_model
from src.utils.pickle import read_configs


class GenericDataModule:
    def __init__(self, dataset_config, out_features, batch_size, seed,
                 device, num_workers=0, target_transform=None):
        self.dataset_config = dataset_config
        self.out_features = out_features
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.device = device
        self.target_transform = target_transform
        self.df = pd.read_csv(dataset_config.labels_path)
        self.tr = None
        self.val = None
        self.test = None

######## handles splitting ########
    def _base_pools(self):
        """Set self.tr / self.val / self.test to index arrays.
        Depends on the specific dataset
        (some have predefined train/val/test/spltits,
        others dont, some have nans others dont..."""
        raise NotImplementedError


    def set_default_splits(self, tr_size, val_size, test_size, verbose=True):
        train_pool, test_pool = self._base_pools()

        self.test = balanced_sample(self, test_pool, test_size)
        self.test_size = len(self.test)

        pool = balanced_sample(self, train_pool, tr_size + val_size)
        labels = self.df.loc[pool, self.out_features[0]]
        self.tr, self.val = train_test_split(
            pool, test_size=val_size / (tr_size + val_size),
            stratify=labels, random_state=self.seed)

        if verbose:
            print(f'[INFO] train {len(self.tr)}  val {len(self.val)}  test {len(self.test)}')
            self._print_balance()

    def _print_balance(self):
        for name, idx in [('train', self.tr), ('val', self.val), ('test', self.test)]:
            if idx is None or len(idx) == 0:
                continue
            counts = self.df.loc[idx, self.out_features].mean().round(3).to_dict()
            print(f'[INFO] {name}: positive rate {counts}')



######## data loading and preparation ########
    def _transform(self, stage):
        t = self.dataset_config.transform
        if isinstance(t, dict):
            return t['fit'] if stage in ('fit', 'train', 'trainval') else t['predict']
        return t

    def _filter_conditionals(self, index, conditionals):
        if not conditionals:
            return index
        sub = self.df.loc[index]
        mask = pd.Series(True, index=sub.index)
        for value, column in conditionals:
            mask &= (sub[column] == value)
        filtered = sub.index[mask.values].to_numpy()
        if len(filtered) == 0:
            raise ValueError(f"No samples satisfy conditionals: {conditionals}")
        return filtered

    def _cf_sample(self, concept, mode=None, conditionals=None):
        rng = np.random.default_rng(self.seed)
        idx = self._filter_conditionals(self.df.index.to_numpy(), conditionals)
        sub = self.df.loc[idx]
        size = int(self.test_size / 2) # 6

        if mode is not None:
            pool = sub.index[sub[concept] == mode].to_numpy()
            if size >= len(pool):
                return pool
            return rng.choice(pool, size=size, replace=False)

        pos = sub.index[sub[concept] == 1].to_numpy()
        neg = sub.index[sub[concept] == 0].to_numpy()
        per = min(size // 2, len(pos), len(neg))
        pos_sel = rng.choice(pos, size=per, replace=False)
        neg_sel = rng.choice(neg, size=per, replace=False)
        return np.concatenate([pos_sel, neg_sel])


    def get_dataset(self, stage, index_split=None):
        temp_df = self.df if index_split is None else self.df.loc[index_split]
        return GenericDataset(temp_df.reset_index(drop=True),
                              self.dataset_config.load_input,
                              self._transform(stage),
                              self.out_features,
                              self.dataset_config,
                              self.target_transform)

    def _get_dataloader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    def train_dataloader(self):
        return self._get_dataloader(self.get_dataset('fit', self.tr), shuffle=True)

    def val_dataloader(self):
        return self._get_dataloader(self.get_dataset('predict', self.val), shuffle=False)

    def test_dataloader(self):
        return self._get_dataloader(self.get_dataset('predict', self.test), shuffle=False)

    def predict_dataloader(self):
        return self._get_dataloader(self.get_dataset('predict', self.df.index), shuffle=False)

    def get_x_y(self, stage, device=None):
        loaders = {
            'train':    self.train_dataloader,
            'validate': self.val_dataloader,
            'test':     self.test_dataloader,
            'predict':  self.predict_dataloader,
        }

        dl = loaders[stage]()
        xs, ys = zip(*dl)
        x = torch.cat(xs, dim=0).detach()
        y = torch.cat(ys, dim=0).detach()
        if device is not None:
            x, y = x.to(device), y.to(device)
        return x, y

######## counterfactual generation ########
def get_counterfactual_pairs(self, concept, conditionals=None):
    """Return (names, x_original, x_counterfactual) for an attribute intervention.
    Depends on the realization function of each dataset"""
    raise NotImplementedError

class GenericProbingDataModule(GenericDataModule):
    def __init__(self, dataset_config, out_features, batch_size,
                 seed, device, main_config, to_extract=None):
        super().__init__(
            dataset_config=dataset_config,
            out_features=out_features,
            batch_size=batch_size,
            seed=seed,
            device=device,)

        self.main_config = main_config
        if to_extract is None:
            self.to_extract = main_config.layers
        model = load_model(main_config.path, self.device)
        self.extractor = ActivationsExtractor(model, self.to_extract)

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
        x, y = zip(*batch)

        x = torch.stack([
            t if torch.is_tensor(t) else torch.as_tensor(t)     # needed to do this also to x bcause of adult dataset
            for t in x
        ]).to(self.device)

        y = torch.stack([
            t.clone().detach() if isinstance(t, torch.Tensor) else torch.tensor(t)
            for t in y
        ]).to(self.device)

        acts = self.extractor.extract(x, as_numpy=False, flatten=True)

        return acts, y

    def get_activations_counterfactual_pairs(self, concept, conditionals=None):
        names, x, x_cf = self.get_counterfactual_pairs(concept, conditionals)
        x_acts = self.extractor.extract(x, as_numpy=False, flatten=True)
        x_acts_cf = self.extractor.extract(x_cf, as_numpy=False, flatten=True)
        return names, x_acts, x_acts_cf

######## Dataset-specific datamodules ####

class GTSRBDataModule(GenericProbingDataModule):
    def _base_pools(self):
        idx = self.df.index.to_numpy()
        train_pool = idx[self.df['train_data'] == 1]
        test_pool = idx[self.df['train_data'] == 0]
        return train_pool, test_pool

    def get_counterfactual_pairs(self, concept, conditionals=None):
        reg = GtsrbInterventionRegistry()
        transform = self._transform('predict')
        load_input = self.dataset_config.load_input
        name_col = self.dataset_config.img_path_col
        imgs_path = self.dataset_config.images_path

        annotated = reg.polygons.get(concept, {})
        cand_names = [n for n, p in annotated.items() if p]
        cand_idx = self.df.index[self.df[name_col].isin(cand_names)].to_numpy()
        cand_idx = self._filter_conditionals(cand_idx, conditionals)
        allowed = set(self.df.loc[cand_idx, name_col])

        names, orig, cf = [], [], []
        for name, polys in annotated.items():
            if not polys or name not in allowed:
                continue
            row_i = self.df.index[self.df[name_col] == name][0]
            x_orig = transform(load_input(self.df, row_i, self.dataset_config))
            image = np.array(Image.open(imgs_path / name).convert('RGB'))
            edited = reg.img_result(image, name, concept)
            names.append(name)
            orig.append(x_orig)
            cf.append(transform(Image.fromarray(edited)))
        if not orig:
            return [], torch.empty(0), torch.empty(0)
        return names, torch.stack(orig), torch.stack(cf)



class CUBDataModule(GenericProbingDataModule):
    def _base_pools(self):
        idx = self.df.index.to_numpy()
        train_pool = idx[self.df['train_data'] == 1]
        test_pool = idx[self.df['train_data'] == 0]
        return train_pool, test_pool

    def get_counterfactual_pairs(self, concept, conditionals=None):
        reg = CubInterventionRegistry()
        transform = self._transform('predict')
        load_input = self.dataset_config.load_input
        name_col = self.dataset_config.img_path_col
        imgs_path = self.dataset_config.images_path

        samples = self._cf_sample(concept, 1, conditionals)

        names, orig, cf = [], [], []
        for i in samples:
            name = self.df[name_col].iloc[i]
            image = np.array(Image.open(imgs_path / name).convert('RGB'))
            edited = reg.img_result(image, name, concept)
            if np.array_equal(edited, image):
                continue
            names.append(name)
            orig.append(transform(load_input(self.df, i, self.dataset_config)))
            cf.append(transform(Image.fromarray(edited)))
        if not orig:
            return [], torch.empty(0), torch.empty(0)
        return names, torch.stack(orig), torch.stack(cf)


class AdultDataModule(GenericProbingDataModule):
    def _base_pools(self):
        idx = self.df.index.to_numpy()
        train_pool = idx[self.df['split'] == 'train']
        test_pool = idx[self.df['split'] == 'test']
        return train_pool, test_pool

    def get_counterfactual_pairs(self, concept, conditionals=None):
        reg = AdultInterventionRegistryTabular()
        transform = self.dataset_config.transform


        if concept == "c_investment_activity":
            samples = self._cf_sample(concept, 1, conditionals)
        else:
            samples = self._cf_sample(concept, 0, conditionals)

        rows = self.df.loc[samples]
        flipped = pd.DataFrame([reg.intervene(r, concept) for _, r in rows.iterrows()])
        names = list(rows.index)
        x_orig = torch.tensor(transform(rows))
        x_cf = torch.tensor(transform(flipped))
        return names, x_orig, x_cf



class XTRAINSDataModule(GenericProbingDataModule):
    def _base_pools(self):
        idx = self.df.index.to_numpy()
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(idx)
        n_test = int(0.2 * len(idx))
        return idx[n_test:], idx[:n_test]

    def get_counterfactual_pairs(self, concept, conditionals=None):
        reg = InterventionManagerXTRAINS()
        intervention = reg.default_intervention(concept)
        transform = self._transform('predict')

        samples = self._cf_sample(concept, 1, conditionals)
        names = self.df.loc[samples, self.dataset_config.img_path_col].tolist()
        train_configs = read_configs(self.dataset_config.configs_path,
                                     datasets_path=str(DATASETS_PATH), as_dict=True)
        _configs = filter_configs(train_configs, names)

        orig, cf = [], []
        for name in names:
            config = _configs[name]
            img    = render_in_mem(copy.deepcopy(config))
            img_cf = intervention.img_result(copy.deepcopy(config))
            orig.append(transform(img))
            cf.append(transform(img_cf))
        if not orig:
            return [], torch.empty(0), torch.empty(0)
        return names, torch.stack(orig), torch.stack(cf)


class CelebaDataModule(GenericProbingDataModule):
    def _base_pools(self):
        idx = self.df.index.to_numpy()
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(idx)
        n_test = int(0.2 * len(idx))
        return idx[n_test:], idx[:n_test]

    def get_counterfactual_pairs(self, concept, conditionals=None):
        gen = AttGANModel(self.dataset_config.generator_model_path)
        transform = self._transform('predict')
        name_col = self.dataset_config.img_path_col
        imgs_path = self.dataset_config.images_path

        samples = self._cf_sample(concept, 0, conditionals)

        names, orig, cf = [], [], []
        for i in samples:
            name = self.df[name_col].iloc[i]
            src = self.df.loc[i, gen.ATTRS].tolist()
            path = imgs_path / name
            cf_img = gen.img_result(path, src, concept)
            names.append(name)
            orig.append(transform(Image.open(path).convert('RGB')))

            aux_transform = transforms.Compose([ # to solve the double cropping problem
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

            cf.append(aux_transform(Image.fromarray((cf_img * 255).astype('uint8'))))
        if not orig:
            return [], torch.empty(0), torch.empty(0)
        return names, torch.stack(orig), torch.stack(cf)


