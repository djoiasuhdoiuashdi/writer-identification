import os
from itertools import combinations
from collections import defaultdict
import random
import json
import argparse


def main():
    parser = argparse.ArgumentParser(description="Generate train-test splits for images.")
    parser.add_argument("input_dir", help="Directory containing the image files.")
    args = parser.parse_args()
    input_dir = args.input_dir
    random.seed(42)
    splits = defaultdict(list)
    author_files = defaultdict(list)

    for file in sorted(os.listdir(input_dir)):
        author = file.split("_")[0]  # Extract author ID from filename
        author_files[author].append(file.replace(".tiff", ".npy"))

    for list_of_files in author_files.values():
        combos = combinations(list_of_files, 2)

        for combo in combos:
            train_split = [image for image in combo]
            author = train_split[0].split("_")[0]
            test_split = [f for f in list_of_files if f not in train_split]
            splits[author].append((train_split, test_split))

    final_splits = []
    for i in range(500):
        train_split = []
        test_split = []
        train_labels = []
        test_labels = []
        for author, split_list in splits.items():
            rand_index = random.randint(0, len(split_list) - 1)
            train_split.extend(split_list[rand_index][0])
            test_split.extend(split_list[rand_index][1])
            train_labels.extend([author] * len(split_list[rand_index][0]))
            test_labels.extend([author] * len(split_list[rand_index][1]))

        final_splits.append(
            {"train": train_split, "test": test_split, "train_labels": train_labels, "test_labels": test_labels})

    with open("train_test_splits.json", "w") as f:
        json.dump(final_splits, f, indent=4)

    print(f"Generated {len(final_splits)} train-test splits.")


if __name__ == "__main__":
    main()
