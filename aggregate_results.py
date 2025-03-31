import os
import pandas as pd
import json


def main():
    path = "./evaluation"

    retrieval_table = [
        ["Mohammed et al. [23]", "30", "", "", ""],
        ["SIFT (Baseline)", "28", "70", "84", "30.3"],
        ["Su Binarization + SIFT", "40", "72", "86", "30.5"],
        ["AngU-Net + SIFT", "46", "84", "88", "36.5"],
        ["AngU-Net + R-SIFT", "48", "84", "92", "42.8"],
        ["AngU-Net + Cl-S [10]", "52", "82", "94", "42.2"],
    ]

    identification_table = [
        ["Mohammed et al. [23]", "26", ""],
        ["Nasir & Siddiqi [24]", "54", ""],
        ["Nasir et al. [25]", "64", ""],
        ["AngU-Net + SIFT + NN", "47", "83"],
        ["AngU-Net + SIFT + SVM", "57", "87"],
        ["AngU-Net + R-SIFT + NN", "53", "77"],
        ["AngU-Net + R-SIFT + SVM", "60", "80"],
    ]

    retrieval_df = pd.DataFrame(retrieval_table, columns=["Approach", "Top-1", "Top-5", "Top-10", "mAP"])
    identification_df = pd.DataFrame(identification_table, columns=["Approach", "Top-1", "Top-5"])

    for approach_name in sorted(os.listdir(path)):
        with open(os.path.join(path, approach_name, "Retrieval.txt"), 'r') as f:
            data = json.load(f)
            retrieval_df.loc[len(retrieval_df)] = [
                approach_name,
                data["top1"],
                data["top5"],
                data["top10"],
                data["map"]
            ]
        with open(os.path.join(path, approach_name, "Classification.txt"), 'r') as f:
            data = json.load(f)
            identification_df.loc[len(identification_df)] = [
                approach_name,
                data["top1"],
                data["top5"]
            ]

    print(retrieval_df)
    print(identification_df)
    retrieval_df.to_csv("retrieval.csv", index=False)
    identification_df.to_csv("identification.csv", index=False)


if __name__ == "__main__":
    main()
