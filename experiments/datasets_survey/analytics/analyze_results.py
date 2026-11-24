import pandas as pd

from src.reliability.tnorms import tnorm_min

INPUT_CSV = "results_final.csv"
LINEAR_CSV = "1_linear_by_concept.csv"
ALL_PROBES_CSV = "2_allprobes_by_concept.csv"

KEYS = ["dataset_cfg", "model_cfg", "out_concept", "out_concept_fmt"]
METRICS = ["test_acc_mean", "test_acc_std", "alignment_mean", "alignment_std"]

CONCEPT_FMT = {
    # GTSRB
    "Symbol": r"$\mathsf{\exists hasSymbol.\top}$",
    "Black_Symbol": r"$\mathsf{\exists hasSymbol.Black}$",
    "Black_Bar": r"$\mathsf{\exists hasBar.Black}$",
    "Post": r"$\mathsf{Pole}$",
    "Red_Ground": r"$\mathsf{\exists hasGround.Red}$",
    "Symbol_Speed80": r"$\mathsf{\exists hasSymbol.Speed80}$",
    "White_Ground": r"$\mathsf{\exists hasGround.White}$",

    # CUB
    "attr_4": r"$\mathsf{\exists hasBillShape.Needle}$",
    "attr_5": r"$\mathsf{\exists hasBillShape.HookedSeabird}$",
    "attr_104": r"$\mathsf{\exists hasHeadPattern.Striped}$",
    "attr_112": r"$\mathsf{\exists hasBreastColor.Yellow}$",
    "attr_134": r"$\mathsf{\exists hasThroatColor.Red}$",
    "attr_148": r"$\mathsf{\exists hasEyeColor.Red}$",
    "attr_198": r"$\mathsf{\exists hasBellyColor.Blue}$",
    "attr_204": r"$\mathsf{\exists hasBellyColor.Yellow}$",
    "attr_226": r"$\mathsf{\exists hasShape.DuckLike}$",
    "attr_306": r"$\mathsf{\exists hasCrownColor.White}$",
    "attr_307": r"$\mathsf{\exists hasCrownColor.Red}$",

    # CelebA
    "Blond_Hair": r"$\mathsf{\exists hasHairColor.Blond}$",
    "Brown_Hair": r"$\mathsf{\exists hasHairColor.Brown}$",
    "Black_Hair": r"$\mathsf{\exists hasHairColor.Black}$",
    "Bald": r"$\mathsf{\exists hasHair.\top}$",
    "Bangs": r"$\mathsf{\exists hasHairStyle.Bangs}$",
    "Mustache": r"$\mathsf{\exists hasBeard.Mustache}$",
    "No_Beard": r"$\mathsf{\exists hasBeard.\top}$",
    "Mouth_Slightly_Open": r"$\mathsf{\exists has.MouthOpen}$",
    "Male": r"$\mathsf{Male}$",
    "Young": r"$\mathsf{Young}$",
    "Pale_Skin": r"$\mathsf{\exists hasSkinTone.Pale}$",
    "Bushy_Eyebrows": r"$\mathsf{\exists hasEyebrow.Bushy}$",
    "Eyeglasses": r"$\mathsf{\exists has.Eyeglasses}$",

    # Adult
    "c_advanced_credential": r"$\mathsf{AdvancedCredential}$",
    "c_investment_activity": r"$\mathsf{InvestmentActivity}$",
    "c_prime_working": r"$\mathsf{PrimeWorking}$",
    "c_married_man": r"$\mathsf{MarriedMan}$",
    "c_demographic_majority": r"$\mathsf{DemographicMajority}$",
    "c_is_male": r"$\mathsf{Male}$",
    "c_country_early": r"$\mathsf{LowLexicalCountry}$",
    "c_odd_age": r"$\mathsf{OddAge}$",
    "c_workclass_midlen": r"$\mathsf{LongWorkClassName}$",
}

DS_CONCEPT_INDEP = {
    "gtsrb": {
        "White_Ground": {
            "dep": ["White_Ground"],
            "indep": ["Symbol_Speed80", "Post", "Symbol", "Black_Symbol", "Black_Bar", "Red_Ground"],
        },
        "Symbol_Speed80": {
            "dep": ["Symbol_Speed80"],
            "indep": ["White_Ground", "Post", "Black_Bar", "Red_Ground"],
        },
        "Post": {
            "dep": ["Post"],
            "indep": ["White_Ground", "Symbol_Speed80", "Symbol", "Black_Symbol", "Black_Bar", "Red_Ground"],
        },
        "Symbol": {
            "dep": ["Symbol_Speed80", "Symbol", "Black_Symbol"],
            "indep": ["White_Ground", "Post", "Red_Ground"],
        },
        "Black_Symbol": {
            "dep": ["Black_Symbol", "Symbol_Speed80"],
            "indep": ["White_Ground", "Post", "Red_Ground"],
        },
        "Black_Bar": {
            "dep": ["Black_Bar"],
            "indep": ["White_Ground", "Symbol_Speed80", "Post", "Red_Ground"],
        },
        "Red_Ground": {
            "dep": ["Red_Ground"],
            "indep": ["White_Ground", "Symbol_Speed80", "Post", "Symbol", "Black_Symbol", "Black_Bar"],
        },
    },
    "cub": {
        "attr_4": {
            "dep": ["attr_4"],
            "indep": ["attr_104", "attr_112", "attr_134", "attr_148", "attr_198", "attr_204", "attr_306", "attr_307"],
        },
        "attr_5": {
            "dep": ["attr_5"],
            "indep": ["attr_104", "attr_112", "attr_134", "attr_148", "attr_198", "attr_204", "attr_306", "attr_307"],
        },
        "attr_104": {
            "dep": ["attr_104"],
            "indep": ["attr_4", "attr_5", "attr_112", "attr_134", "attr_148", "attr_198", "attr_204",
                      "attr_306",
                      "attr_307"],
        },
        "attr_112": {
            "dep": ["attr_112"],
            "indep": ["attr_4", "attr_5", "attr_104", "attr_134", "attr_148", "attr_198",
                      "attr_204",
                      "attr_306", "attr_307"],
        },
        "attr_134": {
            "dep": ["attr_134"],
            "indep": ["attr_4", "attr_5", "attr_104", "attr_112", "attr_148", "attr_198", "attr_204", "attr_306", "attr_307"],
        },
        "attr_148": {
            "dep": ["attr_148"],
            "indep": ["attr_4", "attr_5", "attr_104", "attr_112", "attr_134", "attr_198", "attr_204", "attr_306", "attr_307"],
        },
        "attr_198": {
            "dep": ["attr_198"],
            "indep": ["attr_4", "attr_5", "attr_104", "attr_112", "attr_134", "attr_148", "attr_204", "attr_306", "attr_307"],
        },
        "attr_204": {
            "dep": ["attr_204"],
            "indep": ["attr_4", "attr_5", "attr_104",
                      "attr_112",
                      "attr_134", "attr_148", "attr_198", "attr_306", "attr_307"],
        },
        "attr_306": {
            "dep": ["attr_306"],
            "indep": ["attr_4", "attr_5", "attr_104", "attr_112", "attr_134", "attr_148", "attr_198", "attr_204",
                      "attr_307"],
        },
        "attr_307": {
            "dep": ["attr_307"],
            "indep": ["attr_4", "attr_5", "attr_104", "attr_112", "attr_134", "attr_148", "attr_198", "attr_204",
                      "attr_306"],
        },
    },
    "celeba": {
        "Blond_Hair": {
            "dep": ["Blond_Hair"],
            "indep": ["Bangs", "Mustache", "No_Beard", "Mouth_Slightly_Open", "Male", "Young", "Pale_Skin",
                      "Bushy_Eyebrows", "Eyeglasses"],
        },
        "Brown_Hair": {
            "dep": ["Brown_Hair"],
            "indep": ["Bangs", "Mustache", "No_Beard", "Mouth_Slightly_Open", "Male", "Young", "Pale_Skin",
                      "Bushy_Eyebrows", "Eyeglasses"],
        },
        "Black_Hair": {
            "dep": ["Black_Hair"],
            "indep": ["Mustache", "No_Beard", "Mouth_Slightly_Open", "Male", "Young", "Pale_Skin",
                      "Bushy_Eyebrows", "Eyeglasses"],
        },
        "Bald": {
            "dep": ["Bald"],
            "indep": ["Blond_Hair", "Brown_Hair", "Black_Hair", "Bangs", "Mustache", "No_Beard",
                      "Mouth_Slightly_Open", "Male", "Young", "Pale_Skin", "Bushy_Eyebrows", "Eyeglasses"],
        },
        "Bangs": {
            "dep": ["Bangs"],
            "indep": ["Blond_Hair", "Brown_Hair", "Black_Hair", "Mustache", "No_Beard", "Mouth_Slightly_Open",
                      "Male", "Young", "Pale_Skin", "Bushy_Eyebrows", "Eyeglasses"],
        },
        "Mustache": {
            "dep": ["Mustache"],
            "indep": ["Blond_Hair", "Brown_Hair", "Black_Hair", "Bald", "Bangs", "Mouth_Slightly_Open",
                      "Young", "Pale_Skin", "Bushy_Eyebrows", "Eyeglasses"],
        },
        "No_Beard": {
            "dep": ["No_Beard"],
            "indep": ["Blond_Hair", "Brown_Hair", "Black_Hair", "Bald", "Bangs", "Mouth_Slightly_Open",
                      "Young", "Pale_Skin", "Bushy_Eyebrows", "Eyeglasses"],
        },
        "Mouth_Slightly_Open": {
            "dep": ["Mouth_Slightly_Open"],
            "indep": ["Blond_Hair", "Brown_Hair", "Black_Hair", "Bald", "Bangs", "Mustache", "No_Beard",
                      "Male", "Young", "Pale_Skin", "Bushy_Eyebrows", "Eyeglasses"],
        },
        "Male": {
            "dep": ["Male"],
            "indep": ["Blond_Hair", "Brown_Hair", "Black_Hair", "Bald", "Bangs", "Mouth_Slightly_Open",
                      "Young", "Pale_Skin", "Bushy_Eyebrows", "Eyeglasses"],
        },
        "Young": {
            "dep": ["Young"],
            "indep": ["Blond_Hair", "Brown_Hair", "Black_Hair", "Bangs", "Mouth_Slightly_Open", "Male",
                      "Pale_Skin", "Bushy_Eyebrows", "Eyeglasses"],
        },
        "Pale_Skin": {
            "dep": ["Pale_Skin"],
            "indep": ["Blond_Hair", "Brown_Hair", "Black_Hair", "Bald", "Bangs", "Mustache", "No_Beard",
                      "Mouth_Slightly_Open", "Male", "Young", "Bushy_Eyebrows", "Eyeglasses"],
        },
        "Bushy_Eyebrows": {
            "dep": ["Bushy_Eyebrows"],
            "indep": ["Blond_Hair", "Brown_Hair", "Black_Hair", "Bald", "Bangs", "Mustache", "No_Beard",
                      "Mouth_Slightly_Open", "Young", "Pale_Skin", "Eyeglasses"],
        },
        "Eyeglasses": {
            "dep": ["Eyeglasses"],
            "indep": ["Blond_Hair", "Brown_Hair", "Black_Hair", "Bald", "Bangs", "Mustache", "No_Beard",
                      "Mouth_Slightly_Open", "Male", "Young", "Pale_Skin", "Bushy_Eyebrows"],
        },
    },
    "adult": {
        "c_advanced_credential": {
            "dep": ["c_advanced_credential"],
            "indep": ["c_investment_activity", "c_prime_working", "c_married_man", "c_demographic_majority",
                      "c_is_male", "c_country_early", "c_odd_age", "c_workclass_midlen"],
        },
        "c_investment_activity": {
            "dep": ["c_investment_activity"],
            "indep": ["c_advanced_credential", "c_prime_working", "c_married_man", "c_demographic_majority",
                      "c_is_male", "c_country_early", "c_odd_age", "c_workclass_midlen"],
        },
        "c_prime_working": {
            "dep": ["c_prime_working"],
            "indep": ["c_advanced_credential", "c_investment_activity", "c_married_man", "c_demographic_majority",
                      "c_is_male", "c_country_early", "c_odd_age", "c_workclass_midlen"],
        },
        "c_married_man": {
            "dep": ["c_married_man"],
            "indep": ["c_advanced_credential", "c_investment_activity", "c_prime_working", "c_demographic_majority",
                      "c_country_early", "c_odd_age", "c_workclass_midlen"],
        },
        "c_demographic_majority": {
            "dep": ["c_demographic_majority"],
            "indep": ["c_advanced_credential", "c_investment_activity", "c_prime_working", "c_married_man",
                      "c_is_male", "c_odd_age", "c_workclass_midlen"],
        },
        "c_odd_age": {
            "dep": ["c_odd_age"],
            "indep": ["c_advanced_credential", "c_investment_activity", "c_prime_working", "c_married_man",
                      "c_demographic_majority", "c_is_male", "c_country_early", "c_workclass_midlen"],
        },
    },
}


def compute_alignment(df_concept, spec):
    is_dep = df_concept["protected_concept"].isin(spec["dep"])
    is_indep = df_concept["protected_concept"].isin(spec["indep"])

    dep_values = df_concept.loc[is_dep, "sensitivity_abs"]
    indep_values = df_concept.loc[is_indep, "independence_abs"]

    values = pd.concat([dep_values, indep_values])
    # print("min", tnorm_min(values))
    # print(values, "\n")
    return tnorm_min(values)


def summarize_dataset(df, dataset, probe):
    df = df[(df["dataset_cfg"] == dataset) & (df["probe"] == probe)]

    rows = []
    for model_cfg, df_model in df.groupby("model_cfg"):
        for out_concept, spec in DS_CONCEPT_INDEP[dataset].items():

            acc_values = []
            alignment_values = []
            for seed, df_seed in df_model.groupby("seed"):
                df_concept = df_seed[df_seed["out_concept"] == out_concept]
                if df_concept.empty:
                    continue
                acc_values.append(df_concept["test_acc"].mean())
                # print("out_concept", out_concept)
                alignment_values.append(compute_alignment(df_concept, spec))

            if not acc_values:
                continue

            acc_values = pd.Series(acc_values)
            alignment_values = pd.Series(alignment_values)

            rows.append({
                "dataset_cfg": dataset,
                "model_cfg": model_cfg,
                "out_concept": out_concept,
                "out_concept_fmt": CONCEPT_FMT[out_concept],
                "test_acc_mean": acc_values.mean(),
                "test_acc_std": acc_values.std(),
                "alignment_mean": alignment_values.mean(),
                "alignment_std": alignment_values.std(),
            })

    return pd.DataFrame(rows)


def summarize_all_probes(df):
    per_probe = {}
    for probe in sorted(df["probe"].unique()):
        parts = []
        for dataset in DS_CONCEPT_INDEP:
            parts.append(summarize_dataset(df, dataset, probe))
        per_probe[probe] = pd.concat(parts, ignore_index=True)
        print(f"probe={probe}: {len(per_probe[probe])} rows")
    return per_probe


def combine_pivot(per_probe):
    frames = []
    for probe, df_probe in per_probe.items():
        df_probe = df_probe.set_index(KEYS)[METRICS]
        df_probe.columns = pd.MultiIndex.from_product([[probe], METRICS], names=["probe", "metric"])
        frames.append(df_probe)
    return pd.concat(frames, axis=1)


def main():
    df = pd.read_csv(INPUT_CSV)
    per_probe = summarize_all_probes(df)

    per_probe["linear"].to_csv(LINEAR_CSV, index=False)
    combine_pivot(per_probe).to_csv(ALL_PROBES_CSV)


if __name__ == "__main__":
    main()