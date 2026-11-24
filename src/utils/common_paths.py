from pathlib import Path

PROJECT_PATH = Path(__file__).resolve().parent.parent.parent

OUTPUT_PATH = PROJECT_PATH / 'output'
ASSETS_PATH =  Path("/home/afonso/Desktop/thesis_code/on_the_semantic_reliability") / "assets" # PROJECT_PATH / 'assets'
MAIN_NNS_PATH = ASSETS_PATH / 'main_nns'
DATASETS_PATH = ASSETS_PATH / 'datasets'

XTRAINS_CSV_PATH = DATASETS_PATH / "xtrains_original/csv/labels.csv"
XTRAINS_IMAGE_PATH = DATASETS_PATH / "xtrains_original/images"

XTRAINS_NEW_CSV_PATH = DATASETS_PATH / "xtrains_new/labels.csv"
XTRAINS_NEW_IMAGE_PATH = DATASETS_PATH / "xtrains_new/images"
XTRAINS_NEW_CONFIG_PATH = DATASETS_PATH / "xtrains_new/configs.pickle"
XTRAINS_NEW_CONFIG_DICT_PATH = DATASETS_PATH / "xtrains_new/configs_dict.pickle"

GENERATOR_OUTPUT = Path("/home/afonso/Desktop/thesis_code/other_repos/XTRAINS_generator/output")


GTSRB_CSV_PATH = DATASETS_PATH / "gtsrb/csv/GT_standard.csv"
GTSRB_IMAGE_PATH = DATASETS_PATH / "gtsrb/images"
GTSRB_PARTS_PATH = DATASETS_PATH / "gtsrb/attribute_parts.json" # TODO


CUB_CSV_PATH = DATASETS_PATH / "CUB_200_2011/csv/CUB_attributes_id.csv"
CUB_IMAGE_PATH = DATASETS_PATH / "CUB_200_2011/images"
CUB_PARTS_PATH = DATASETS_PATH / "CUB_200_2011/parts/part_locs.txt"
CUB_BB_PATH = DATASETS_PATH / "CUB_200_2011/bounding_boxes.txt"

CELEBA_CSV_PATH =  DATASETS_PATH / "celebA/csv/labels.csv"
CELEBA_IMAGE_PATH = DATASETS_PATH / "celebA/img_align_celeba/img_align_celeba"
CELEBA_ATT_GENERATOR_PATH = ASSETS_PATH / "generators" / "attgan.pth"

ADULT_CSV_PATH = DATASETS_PATH / "adult/adult_clean.csv"
ADULT_IMAGE_PATH = DATASETS_PATH / "adult/adult_cleab.csv"
ADULT_ENCODINGS_PATH = DATASETS_PATH / "adult/adult_encodings.json"
ADULT_PREP_PATH = DATASETS_PATH / "adult/adult_prep.json"


