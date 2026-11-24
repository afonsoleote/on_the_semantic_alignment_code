from dataclasses import dataclass
from typing import Type, Optional, Callable

from torchvision import transforms

from configs.main_model import TypeAConfig, TypeBConfig, TypeCConfig, ClassIDConfig, SpeciesConfig, AttractiveConfig, \
    IncomeConfig
from scripts.preliminaries_adult.main_income import AdultPreprocessor
from src.data.dataframe_data_handling import load_image_input, load_row_input, load_image_input_cub
from src.data.datamodules_xtrains import ProbingDataModuleXTRAINS
from src.utils.common_paths import XTRAINS_NEW_CSV_PATH, XTRAINS_NEW_IMAGE_PATH, XTRAINS_NEW_CONFIG_PATH, \
    GTSRB_CSV_PATH, GTSRB_IMAGE_PATH, CUB_CSV_PATH, CUB_IMAGE_PATH, CUB_BB_PATH, CUB_PARTS_PATH, CELEBA_CSV_PATH, \
    CELEBA_IMAGE_PATH, CELEBA_ATT_GENERATOR_PATH, ADULT_CSV_PATH, ADULT_IMAGE_PATH, ADULT_PREP_PATH, GTSRB_PARTS_PATH


@dataclass(frozen=True)
class DatasetConfig:
    """Base class for a dataset configuration."""
    id: str
    data_module: Type[ProbingDataModuleXTRAINS] # TODO
    labels_path: str
    images_path: str
    models: tuple
    img_path_col: Optional[str] = None
    data_shape:  Optional[tuple] = None
    configs_path: Optional[str] = None
    bb_path: Optional[str] = None
    bb_parts_path: Optional[str] = None
    generator_model_path: Optional[str] = None
    symbolic_encodings_path: Optional[str] = None
    symbolic_prep_path: Optional[str] = None
    transform: Optional[Callable] = None
    load_input: Optional[Callable] = None

    def __repr__(self):
        return self.id


# configuration for the Explainable Abstract Trains (XTRAINS) dataset
XTRAINSConfig = DatasetConfig(
    id='xtrains',
    img_path_col='image_id',
    data_module=ProbingDataModuleXTRAINS,
    labels_path=XTRAINS_NEW_CSV_PATH,
    images_path=XTRAINS_NEW_IMAGE_PATH,
    models=(TypeAConfig, TypeBConfig, TypeCConfig,),
    data_shape=(3, 152, 152),
    configs_path=XTRAINS_NEW_CONFIG_PATH,
    transform=transforms.Compose([transforms.ToTensor(),]),
    load_input=load_image_input,
)

GTSRBConfig = DatasetConfig(
    id='gtsrb',
    img_path_col='Filename',
    data_module=ProbingDataModuleXTRAINS, #GTSRBDataModule,
    labels_path=GTSRB_CSV_PATH,
    images_path=GTSRB_IMAGE_PATH,
    models=(ClassIDConfig,),
    data_shape=(3, 128, 128),
    bb_parts_path = GTSRB_PARTS_PATH,
    transform= transforms.Compose([
                           transforms.Resize(128),
                           transforms.CenterCrop(128),
                           transforms.ToTensor(),
                       ]),
    load_input=load_image_input,

)

CUBConfig = DatasetConfig(
    id='cub',
    img_path_col='filepath',
    data_module=ProbingDataModuleXTRAINS, #CUBDataModule,
    labels_path=CUB_CSV_PATH,
    images_path=CUB_IMAGE_PATH,
    models=(SpeciesConfig,),
    data_shape=(3, 224, 224),
    bb_path=CUB_BB_PATH,
    bb_parts_path=CUB_PARTS_PATH,
    transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ]),
    load_input = load_image_input_cub,
)

CelebaConfig = DatasetConfig(
    id='celeba',
    img_path_col='image_id',
    data_module=ProbingDataModuleXTRAINS, #CelebaDataModule,
    labels_path=CELEBA_CSV_PATH,
    images_path=CELEBA_IMAGE_PATH,
    models=(AttractiveConfig,),
    data_shape=(3, 128, 128),
    generator_model_path=CELEBA_ATT_GENERATOR_PATH,
    transform=transforms.Compose([
        transforms.Resize(190),
        transforms.CenterCrop(170),
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ]),
    load_input =load_image_input,
)


AdultConfig = DatasetConfig(
    id='adult',
    data_module=ProbingDataModuleXTRAINS, #AdultDataModule,
    labels_path=ADULT_CSV_PATH,
    images_path=ADULT_IMAGE_PATH,
    models = (IncomeConfig, ),
    transform=AdultPreprocessor.load(ADULT_PREP_PATH).transform,
    load_input = load_row_input,
)


DATASET_BY_ID = {
    'xtrains': XTRAINSConfig,
    'gtsrb': GTSRBConfig,
    'cub': CUBConfig,
    'celeba': CelebaConfig,
    'adult': AdultConfig,
}


DATASETS = list(DATASET_BY_ID.values())



