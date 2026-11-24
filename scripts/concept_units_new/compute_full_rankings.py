import ast
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict

def ranking_windows(ranking, window_size=3):
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd.")

    radius = window_size // 2
    windows = []

    for i in range(len(ranking)):
        start = max(0, i - radius)
        end = min(len(ranking), i + radius + 1)

        merged = defaultdict(list)

        for d in ranking[start:end]:
            for k, v in d.items():
                merged[k].extend(v)

        windows.append(dict(merged))

    return windows

def cumulative_ranking(ranking):
    merged = defaultdict(list)
    cumulative = []

    for d in ranking:
        for k, v in d.items():
            merged[k].extend(v)

        cumulative.append({k: list(v) for k, v in merged.items()})

    return cumulative

def merge_ranking(ranking):
    merged = defaultdict(list)

    for item in ranking:
        d = ast.literal_eval(item)
        for k, v in d.items():
            merged[k].extend(v)

    return dict(merged)

def replace_sweep(pc_rank, rc_rank, topk):
    top_pc = pc_rank[:topk]
    top_rc = rc_rank[:topk]

    res = []

    for k in range(topk + 1):
        current = top_pc[:topk - k] + top_rc[:k]

        seen = set()
        current = [x for x in current if not (x in seen or seen.add(x))]

        res.append(merge_ranking(current))
    return res

def interp_sweep(pc, rc, topk):
    res = []

    for alpha in np.linspace(0, 1, topk + 1):
        score = (1 - alpha) * pc + alpha * rc

        ranking = (
            score
            .sort_values(ascending=False)
            .index.tolist()[:topk]
        )

        res.append(merge_ranking(ranking))
    return res


df = pd.read_csv("ranking_full.csv")
df = df[["out_concept", "to_extract", "linear_threshold", "kde_sep"]]

pivot = df.pivot(
    index="to_extract",
    columns="out_concept",
    values=["linear_threshold", "kde_sep"]
)

pivot.columns = [f"{metric}_{concept}" for metric, concept in pivot.columns]


pivot["kde_score"] = (
    pivot["kde_sep_PassengerCar"]
    - pivot["kde_sep_ReinforcedCar"]
)

pivot["kde_score_2"] = (
    (pivot["kde_sep_PassengerCar"]**2) *
    (1- pivot["kde_sep_ReinforcedCar"])
)

pivot["kde_score_3"] = (
    (pivot["kde_sep_PassengerCar"]) /
    (pivot["kde_sep_ReinforcedCar"])
)


kde_ranking = (
    pivot[["kde_score"]]
    .sort_values("kde_score", ascending=False)
)

kde_ranking_2 = (
    pivot[["kde_score_2"]]
    .sort_values("kde_score_2", ascending=False)
)

kde_ranking_3 = (
    pivot[["kde_score_3"]]
    .sort_values("kde_score_3", ascending=False)
)

print("\n=== KDE Separation Ranking ===")
print(kde_ranking)


comparison = pd.DataFrame({
   "kde_rank": kde_ranking.reset_index().reset_index().set_index("to_extract")["index"] + 1,
   "kde_rank2": kde_ranking_2.reset_index().reset_index().set_index("to_extract")["index"] + 1,
   "kde_rank3": kde_ranking_3.reset_index().reset_index().set_index("to_extract")["index"] + 1,
})

print(comparison)


kde_list = [ast.literal_eval(x) for x in kde_ranking.index.tolist()]
kde_list2 = [ast.literal_eval(x) for x in kde_ranking_2.index.tolist()]
kde_list3 = [ast.literal_eval(x) for x in kde_ranking_3.index.tolist()]


rankings = {
    # "kde_w_7": ranking_windows(kde_list, 7),
    # "kde_w_9": ranking_windows(kde_list, 9),
    # "kde_w_11": ranking_windows(kde_list, 11),
    # "kde_w_5_rank2": ranking_windows(kde_list2, 5),
    # "kde_w_5_rank3": ranking_windows(kde_list3, 5),

}

##### this is just to debug #####
pc_rank = (
    pivot["kde_sep_PassengerCar"]
    .sort_values(ascending=False)
    #.index.tolist()
)

rc_rank = (
    pivot["kde_sep_ReinforcedCar"]
    .sort_values(ascending=False)
    #.index.tolist()
)

print(pc_rank)
print(rc_rank)
##### this is just to debug (end) #####

pc_rank = pivot["kde_sep_PassengerCar"].sort_values(ascending=False).index.tolist()
rc_rank = pivot["kde_sep_ReinforcedCar"].sort_values(ascending=False).index.tolist()

rankings["replace_sweep_10_full"] = replace_sweep(pc_rank, rc_rank, 10)
rankings["replace_sweep_15_full"] = replace_sweep(pc_rank, rc_rank, 15)
rankings["replace_sweep_20_full"] = replace_sweep(pc_rank, rc_rank, 20)
rankings["replace_sweep_25_full"] = replace_sweep(pc_rank, rc_rank, 25)
rankings["replace_sweep_30_full"] = replace_sweep(pc_rank, rc_rank, 30)

pc = pivot["kde_sep_PassengerCar"]
rc = pivot["kde_sep_ReinforcedCar"]

rankings["interp_sweep_10_full"] = interp_sweep(pc,rc, 10)
rankings["interp_sweep_15_full"] = interp_sweep(pc,rc, 15)
rankings["interp_sweep_20_full"] = interp_sweep(pc,rc, 20)
rankings["interp_sweep_25_full"] = interp_sweep(pc,rc, 25)
rankings["interp_sweep_30_full"] = interp_sweep(pc,rc, 30)


with open("rankings_full.pickle", "wb") as f:
    pickle.dump(rankings, f)

with open("rankings_full.pickle", "rb") as f:
    rankings = pickle.load(f)

print(rankings.keys())
for r in rankings:
    print(r, len(rankings[r]), rankings[r])