import json
import os
import shutil
import subprocess
import sys
import cv2
import numpy as np
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity
import extract_patches
import vlad
import resnet
from sklearn.neighbors import KNeighborsClassifier
import torchvision.transforms as transforms
import torch
import retrieval_eval

import argparse
from utils import get_author

WINDOW_SIZE = 32
VLAD_NUM_CLUSTER = 128
RESNET_NUM_CLUSTER = 5000
WHITE_PIXEL_THRESHOLD = 0.95
BLACK_PIXEL_THRESHOLD = -1


def get_patch(img, px, py):
    half_win_size = int(WINDOW_SIZE / 2)
    if not (half_win_size < px < img.shape[1] - half_win_size and half_win_size < py < img.shape[0] - half_win_size):
        return None

    roi = img[py - half_win_size:py + half_win_size, px - half_win_size:px + half_win_size]
    assert roi.shape == (half_win_size * 2, half_win_size * 2), 'shape of the roi is not (%d,%d). It is (%d,%d)' % \
                                                                (half_win_size * 2, half_win_size * 2,
                                                                 roi.shape[0], roi.shape[1])
    return roi


def extract_patches_vlad(net, image_path, device):
    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = gray_img.copy()
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    sift = cv2.SIFT_create(sigma=1.6)
    key_pts = sift.detect(gray_img, None)
    num_pix = WINDOW_SIZE ** 2

    patches, seen = [], set()
    for kp in key_pts:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        patch_gray = get_patch(gray_img, x, y)
        if patch_gray is None or (x, y) in seen:
            continue

        if img_bin[y, x] != 0:
            continue

        if extract_patches.black_pixels(patch_gray) > (
                BLACK_PIXEL_THRESHOLD * num_pix) and BLACK_PIXEL_THRESHOLD != -1 or extract_patches.white_pixels(
            patch_gray) > (WHITE_PIXEL_THRESHOLD * num_pix):
            continue

        seen.add((x, y))
        patches.append(patch_gray)

    features = []
    net.linear = torch.nn.Identity()
    transform = transforms.ToTensor()
    with torch.no_grad():
        for i in range(0, len(patches), 64):
            batch_patches = patches[i:i + 64]
            batch_tensor = torch.stack([transform(p) for p in batch_patches]).to(device)
            batch_output = net(batch_tensor)
            features.append(batch_output.cpu())
    features = torch.cat(features, dim=0)
    return features


def main():
    parser = argparse.ArgumentParser(description="Process subfolder")
    parser.add_argument("directory", help="Directory to process")
    args = parser.parse_args()
    directory = args.directory
    path = "./extract_patches_input"

    # ---------------------------------------------------------------------------------------------------
    # EXTRACT PATCHES
    # ---------------------------------------------------------------------------------------------------
    print("Extracting Patches: ", directory)
    input_path = os.path.join(path, directory)
    output_path = os.path.join("extract_patches_output", directory)
    resnet_output_path = os.path.join("resnet20_output", directory)

    if not os.path.exists("extract_patches_output"):
        os.mkdir("extract_patches_output")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    else:
        shutil.rmtree(output_path)
        os.mkdir(output_path)
    subprocess.run(
        [sys.executable, 'extract_patches.py', "--in_dir", input_path, "--out_dir", output_path, "--num_of_clusters",
         f"{RESNET_NUM_CLUSTER}", "--centered", "True", "--black_pixel_thresh", f"{BLACK_PIXEL_THRESHOLD}",
         "--white_pixel_thresh", f"{WHITE_PIXEL_THRESHOLD}", "--scale", "1"]
        , stdout=None, stderr=None)
    center_path = os.path.join(output_path, 'centers.pkl')
    parameter_path = os.path.join(output_path, 'db-creation-parameters.json')
    if os.path.exists(center_path):
        os.remove(center_path)
    if os.path.exists(parameter_path):
        os.remove(parameter_path)

    # ---------------------------------------------------------------------------------------------------
    # TRAIN RESNET
    # ---------------------------------------------------------------------------------------------------
    print("Training Resnet: ", directory)
    if not os.path.exists("resnet20_output"):
        os.mkdir("resnet20_output")
    if not os.path.exists(resnet_output_path):
        os.mkdir(resnet_output_path)

    subprocess.run([sys.executable, 'train_resnet20.py',
                    "--arch", "resnet20",
                    "--workers", "8",
                    "--epochs", "2",
                    "--batch-size", "32",
                    "--lr", "0.01",
                    "--momentum", "0.95",
                    "--weight-decay", "0.00065",
                    "--output_dir", resnet_output_path,
                    "--input_dir", output_path],
                   stdout=None, stderr=None)

    # ---------------------------------------------------------------------------------------------------
    # TRAIN VLAD
    # ---------------------------------------------------------------------------------------------------
    print("Training VLAD: ", directory)
    vlad_train_output_path = os.path.join("vlad_train_output", directory)
    if not os.path.exists("vlad_train_output"):
        os.makedirs("vlad_train_output")
    if not os.path.exists(vlad_train_output_path):
        os.makedirs(vlad_train_output_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = resnet.__dict__["resnet20"](num_classes=RESNET_NUM_CLUSTER)
    checkpoint = torch.load(os.path.join(resnet_output_path, "model.th"), map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    fixed_state_dict = OrderedDict(
        (k.replace("module.", "") if k.startswith("module.") else k, v)
        for k, v in state_dict.items()
    )
    net.load_state_dict(fixed_state_dict, strict=True)
    net = net.to(device).eval()

    vlad_inputs = []
    for input_image in sorted(os.listdir(path=os.path.join(path, directory))):
        print(f"Extracting Features for {input_image}")
        features = extract_patches_vlad(net, os.path.join(path, directory, input_image), device)
        if features is not None:
            vlad_inputs.append(features)

    vlad_inputs = np.vstack(vlad_inputs)
    VLAD = vlad.VLAD(n_clusters=VLAD_NUM_CLUSTER, gmp=True, gamma=1000)
    VLAD._train_impl(vlad_inputs)
    VLAD.save(os.path.join(vlad_train_output_path, "model.pkl"))

    # ---------------------------------------------------------------------------------------------------
    # INFERENCE VLAD
    # ---------------------------------------------------------------------------------------------------
    print("Inferencing VLAD: ", directory)
    vlad_inference_output_path = os.path.join("vlad_inference_output", directory)
    if not os.path.exists("vlad_inference_output"):
        os.makedirs("vlad_inference_output")
    if not os.path.exists(vlad_inference_output_path):
        os.makedirs(vlad_inference_output_path)

    VLAD.load(os.path.join(vlad_train_output_path, "model.pkl"))
    for input_image in sorted(os.listdir(path=os.path.join(path, directory))):
        print(f"Inferencing {input_image} with VLAD")
        features = extract_patches_vlad(net,
                                        os.path.join(path, directory, input_image), device)
        vlad_output = VLAD.encode(features)
        np.save(os.path.join(vlad_inference_output_path, input_image.replace("tiff", "") + "npy"), vlad_output)

    stored_encodings = []
    authors = []
    for input_image in sorted(os.listdir(path=os.path.join(path, directory))):
        base_image_path = os.path.join(vlad_inference_output_path, input_image.replace("tiff", "npy"))
        base_image_encoding = np.load(base_image_path)
        base_image_author = get_author(input_image)
        stored_encodings.append(base_image_encoding)
        authors.append(base_image_author)

    stored_encodings = np.vstack(stored_encodings)
    author_to_id = {}
    author_labels = []
    for author in authors:
        if author not in author_to_id:
            author_to_id[author] = len(author_to_id)
        author_labels.append(author_to_id[author])
    retrieval = retrieval_eval.Retrieval()
    res, tes = retrieval.eval(stored_encodings, author_labels)
    print(tes)
    mAP = res["map"] * 100

    # ---------------------------------------------------------------------------------------------------
    # SIMILARITY CALCULATION
    # ---------------------------------------------------------------------------------------------------
    print("Calculating Cosine Distance Similarity: ", directory)
    similarity_output_path = os.path.join("similarity_output", directory)
    if not os.path.exists("similarity_output"):
        os.makedirs("similarity_output")
    if not os.path.exists(similarity_output_path):
        os.makedirs(similarity_output_path)
    evaluation_result_path = os.path.join("evaluation", directory)
    if not os.path.exists("evaluation"):
        os.makedirs("evaluation")
    if not os.path.exists(evaluation_result_path):
        os.makedirs(evaluation_result_path)

    with open(os.path.join(similarity_output_path, "results.txt"), "w") as f:
        stats = []
        for input_image in sorted(os.listdir(path=os.path.join(path, directory))):
            base_image_path = os.path.join(vlad_inference_output_path, input_image.replace("tiff", "npy"))
            base_image_encoding = np.load(base_image_path)
            base_image_author = get_author(input_image)
            stored_encodings = []
            file_paths = []

            for image_to_compare_to in sorted(os.listdir(vlad_inference_output_path)):
                if image_to_compare_to.endswith(".npy") and image_to_compare_to != os.path.basename(
                        base_image_path):
                    file_path = os.path.join(vlad_inference_output_path, image_to_compare_to)
                    stored_encodings.append(np.load(file_path))
                    file_paths.append(image_to_compare_to)

            stored_encodings = np.vstack(stored_encodings)
            similarities = cosine_similarity(base_image_encoding, stored_encodings)[0]
            indices = np.argsort(similarities)[::-1][:10]
            most_similar_files = [file_paths[i].split('_')[0] for i in indices]
            stats.append({
                "author": base_image_author,
                "top1": base_image_author in most_similar_files[:1],
                "top5": base_image_author in most_similar_files[:5],
                "top10": base_image_author in most_similar_files[:10]
            })
        f.write(json.dumps(stats, indent=4))

    # ---------------------------------------------------------------------------------------------------
    # CALCULATE WRITER RETRIEVAL TABLE
    # ---------------------------------------------------------------------------------------------------

    file_path = os.path.join(similarity_output_path, "results.txt")
    with open(file_path, 'r') as file:
        data = json.load(file)

    total = len(data)
    top1_count = sum(1 for result in data if result["top1"])
    top5_count = sum(1 for result in data if result["top5"])
    top10_count = sum(1 for result in data if result["top10"])

    top1_accuracy = (top1_count / total) * 100
    top5_accuracy = (top5_count / total) * 100
    top10_accuracy = (top10_count / total) * 100

    with open(evaluation_result_path + "/Retrieval.txt", "w") as f:
        json.dump({"top1": round(top1_accuracy), "top5": round(top5_accuracy), "top10": round(top10_accuracy),
                   "map": round(mAP)}, f, indent=4)

    # ----------------------------------------------------------------------------------------------------
    # WRITER CLASSIFICATION
    # ----------------------------------------------------------------------------------------------------

    print("Writer Classification: ", directory)
    combinations = []
    with open("./train_test_splits.json", "r") as f:
        combinations = json.load(f)

    split_results = []

    cache = {}
    for file in sorted(os.listdir(vlad_inference_output_path)):
        cache[file] = np.load(os.path.join(vlad_inference_output_path, file))

    splits = []
    for combo in combinations:
        training_set = []
        test_set = []
        test_labels = []
        train_labels = []
        for filepath in combo["train"]:
            training_set.append(cache[filepath])
        for filepath in combo["test"]:
            test_set.append(cache[filepath])
        for label in combo["train_labels"]:
            train_labels.append(label)
        for label in combo["test_labels"]:
            test_labels.append(label)
        splits.append((training_set, train_labels, test_set, test_labels))

    for training_set, training_labels, test_set, test_labels in splits:
        training_set = np.vstack(training_set)
        test_set = np.vstack(test_set)
        nn = KNeighborsClassifier(n_neighbors=5)
        nn.fit(training_set, training_labels)
        distances, indices = nn.kneighbors(test_set, n_neighbors=5)
        training_labels = np.array(training_labels)
        top1_count = 0
        top5_count = 0
        total = len(test_labels)
        for i, neighbor_indices in enumerate(indices):
            top5_predictions = training_labels[neighbor_indices]
            if test_labels[i] == top5_predictions[0]:
                top1_count += 1
                top5_count += 1
            elif test_labels[i] in top5_predictions:
                top5_count += 1
        top1_accuracy = (top1_count / total) * 100
        top5_accuracy = (top5_count / total) * 100
        split_results.append((top1_accuracy, top5_accuracy))
        print(f"Top1: {top1_accuracy}, Top5: {top5_accuracy}")

    avg_top1 = sum(r[0] for r in split_results) / len(split_results)
    avg_top5 = sum(r[1] for r in split_results) / len(split_results)

    with open(evaluation_result_path + "/Classification.txt", "w") as f:
        json.dump({"top1": round(avg_top1), "top5": round(avg_top5)}, f, indent=4)


if __name__ == "__main__":
    main()
