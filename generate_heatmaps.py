import argparse
import itertools
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_author


# ---------------------------------------------------------------------------------------------------
# HEATMAP GENERATION WRITER RETRIEVAL a) and b)
# ---------------------------------------------------------------------------------------------------
def writer_retrieval(approach_name):
    path = os.path.join("./vlad_inference_output", approach_name)
    # ---------------------------------------------------------------------------------
    # a) Retrieval heatmap
    # ---------------------------------------------------------------------------------
    cosine_similarities = []
    for item in os.listdir(path):
        cosine_similarities.append(np.load(os.path.join(path, item)))

    cosine_similarities = np.vstack(cosine_similarities)
    matrix = cosine_similarity(cosine_similarities)
    plt.figure(figsize=(8, 6))

    ax = sns.heatmap(matrix, cmap="crest", square=True, cbar=False)

    ticksX = np.array([0, 20, 40])
    ticksY = np.array([0, 10, 20, 30, 40])
    ax.set_xticks(ticksX + 0.5)
    ax.set_xticklabels(ticksX, rotation=45)
    ax.set_yticks(ticksY + 0.5)
    ax.set_yticklabels(ticksY)
    plt.savefig("retrieval_heatmap_a.png", dpi=300, bbox_inches="tight")

    # ---------------------------------------------------------------------------------------------------
    # b) Retrieval heatmap
    # ---------------------------------------------------------------------------------------------------
    cosine_similarities = []
    authors = set()
    for item in os.listdir(path):
        cosine_similarities.append((np.load(os.path.join(path, item)), item))
        authors.add(get_author(item))

    authors = sorted(list(authors))
    num_authors = len(authors)
    pairs = itertools.combinations(range(len(cosine_similarities)), 2)
    matrix = np.zeros((num_authors, num_authors))
    sim_count = np.zeros((num_authors, num_authors))
    for i, j in pairs:
        author1 = get_author(cosine_similarities[i][1])
        author2 = get_author(cosine_similarities[j][1])

        index1 = authors.index(author1)
        index2 = authors.index(author2)

        matrix[index1][index2] += cosine_similarity(cosine_similarities[i][0].reshape(1, -1),
                                                    cosine_similarities[j][0].reshape(1, -1))[0, 0]

        matrix[index2][index1] += cosine_similarity(cosine_similarities[i][0].reshape(1, -1),
                                                    cosine_similarities[j][0].reshape(1, -1))[0, 0]
        sim_count[index1][index2] += 1
        sim_count[index2][index1] += 1

    avg_sim = np.divide(matrix, sim_count, out=np.zeros_like(matrix), where=sim_count != 0)
    plt.figure(figsize=(6, 8))

    ax = sns.heatmap(avg_sim, cmap="crest", square=True, cbar=False)
    ticksX = np.arange(len(authors))
    ticksY = np.arange(len(authors))
    ax.set_xticks(ticksX + 0.5)
    ax.set_xticklabels(ticksX + 1, rotation=45)
    ax.set_yticks(ticksY + 0.5)
    ax.set_yticklabels([f"{i}: {author}" for i, author in enumerate(authors)], rotation=0)
    plt.savefig("retrieval_heatmap_b.png", dpi=300, bbox_inches="tight")


# ---------------------------------------------------------------------------------------------------
# HEATMAP GENERATION WRITER IDENTIFICATION
# ---------------------------------------------------------------------------------------------------
def writer_identification(approach_name):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate heatmaps for writer retrieval and identification.")
    parser.add_argument("input_dir", help="Directory containing the image files.")
    args = parser.parse_args()
    writer_retrieval(args.input_dir)
    writer_identification(args.input_dir)
