import pandas as pd
import matplotlib.pyplot as plt

INPUT_CSV = "results_processed.csv"

DEP_CONCEPT = "PassengerCar"
INDEP_CONCEPT = "ReinforcedCar"


def main():
    df = pd.read_csv(INPUT_CSV)
    df = df[['corr', 'protected_concept', 'to_extract',
             'sensitivity', 'sensitivity_std', 'independence',
             'independence_std', 'alg', 'alg_std',
             'test_acc', 'test_acc_std']]
    table = df.round(3).astype(object)

    for i in range(len(df)):
        concept = df["protected_concept"][i]
        if concept == DEP_CONCEPT:
            table.at[i, "independence"] = "-"
            table.at[i, "independence_std"] = "-"
        elif concept == INDEP_CONCEPT:
            table.at[i, "sensitivity"] = "-"
            table.at[i, "sensitivity_std"] = "-"

    fig, ax = plt.subplots(figsize=(12,4))
    ax.axis("off")
    t = ax.table(cellText=table.values, colLabels=table.columns, loc="center")
    t.set_fontsize(10)
    t.auto_set_column_width(range(len(table.columns)))

    plt.show()


if __name__ == "__main__":
    main()