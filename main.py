import json
import os
import random
import shutil
from sklearn.metrics import confusion_matrix
import subprocess
import sys
import cv2
import numpy as np
from collections import OrderedDict, defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import extract_patches
import vlad
import resnet
from sklearn.neighbors import KNeighborsClassifier
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.neighbors import NearestNeighbors

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

        if img_bin[y,x] != 0:
            continue

        if extract_patches.black_pixels(patch_gray) > (BLACK_PIXEL_THRESHOLD * num_pix) and BLACK_PIXEL_THRESHOLD != -1 or extract_patches.white_pixels(patch_gray) > (WHITE_PIXEL_THRESHOLD * num_pix):
            continue

        seen.add((x, y))
        patches.append(patch_gray)

    current_out = None

    def hook(module, inp, out):
        nonlocal current_out
        current_out = out.detach()

    net.layer3.register_forward_hook(hook)

    with torch.no_grad():
        if patches:
            all_features = []
            batch_size = 64

            for i in range(0, len(patches), batch_size):
                batch_patches = patches[i:i + batch_size]
                batch_tensor = torch.stack([
                    transforms.ToTensor()(p) for p in batch_patches
                ]).to(device)
                net(batch_tensor)
                current_out = torch.nn.functional.avg_pool2d(current_out, current_out.size()[3])
                batch_features = current_out.view(current_out.size(0), -1)
                all_features.append(batch_features.cpu())

            return np.vstack(all_features)
        return None




def main():

    path = "./extract_patches_input"
    dirs = sorted(os.listdir(path=path))
    for directory in dirs:

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
        subprocess.run([sys.executable, 'extract_patches.py', "--in_dir", input_path, "--out_dir", output_path, "--num_of_clusters", f"{RESNET_NUM_CLUSTER}", "--centered", "True", "--black_pixel_thresh", f"{BLACK_PIXEL_THRESHOLD}", "--white_pixel_thresh", f"{WHITE_PIXEL_THRESHOLD}", "--scale", "1"]
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
            "--epochs", "200",
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
                base_image_author = input_image.split("_")[0]
                stored_encodings = []
                file_paths = []

                for image_to_compare_to in sorted(os.listdir(vlad_inference_output_path)):
                    if image_to_compare_to.endswith(".npy") and image_to_compare_to != os.path.basename(base_image_path):
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

        table_data = [
            ["Mohammed et al. [23]", "30", "", "", ""],
            ["SIFT (Baseline)", "28", "70", "84", "30.3"],
            ["Su Binarization + SIFT", "40", "72", "86", "30.5"],
            ["AngU-Net + SIFT", "46", "84", "88", "36.5"],
            ["AngU-Net + R-SIFT", "48", "84", "92", "42.8"],
            ["AngU-Net + Cl-S [10]", "52", "82", "94", "42.2"],
            [directory, f"{round(top1_accuracy)}", f"{round(top5_accuracy)}", f"{round(top10_accuracy)}", ""]
        ]
        headers = ["Method", "Top-1", "Top-5", "Top-10", "mAP"]
        print(tabulate(table_data, headers, tablefmt="grid"))
        with open(evaluation_result_path + "/Retrieval.txt", "w") as f:
            f.write(tabulate(table_data, headers, tablefmt="grid"))

        # ---------------------------------------------------------------------------------------------------
        # HEATMAP GENERATION TODO: REWRITE
        # ---------------------------------------------------------------------------------------------------
        # print("Generating Heatmaps: ", directory)
        #
        #
        # author_encodings = defaultdict(list)
        # for base_image_path in sorted(os.listdir(vlad_inference_output_path)):
        #     if base_image_path.endswith(".npy"):
        #         file_path = os.path.join(vlad_inference_output_path, base_image_path)
        #         base_image_author = base_image_path.split("_")[0]
        #         encoding = np.load(file_path)
        #         author_encodings[base_image_author].append(encoding)
        #
        # authors = sorted(author_encodings.keys())
        # similarity_matrix = np.zeros((10,10))
        #
        # for i, author1 in enumerate(authors):
        #     for j, author2 in enumerate(authors):
        #         encodings_author1 = np.vstack(author_encodings[author1])
        #         encodings_author2 = np.vstack(author_encodings[author2])
        #         similarities = cosine_similarity(encodings_author1, encodings_author2)
        #         avg_similarity = np.mean(similarities)
        #         similarity_matrix[i, j] = avg_similarity
        #
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(
        #     similarity_matrix,
        #     cmap="Wistia",
        #     xticklabels=authors,
        #     yticklabels=authors,
        #     annot=False,
        #     vmin=0,
        #     vmax=1
        # )
        # plt.title(f'Inter-scribe Similarity Heatmap - {directory}')
        # plt.tight_layout()
        #
        # plt.savefig(os.path.join(evaluation_result_path, "scribe_similarity_heatmap.png"), dpi=300)
        # plt.close()
        #
        #
        # all_encodings = []
        # for base_image_path in sorted(os.listdir(vlad_inference_output_path)):
        #     if base_image_path.endswith(".npy"):
        #         file_path = os.path.join(vlad_inference_output_path, base_image_path)
        #         encoding = np.load(file_path)
        #         all_encodings.append(encoding)
        #
        #
        # all_encodings = np.vstack(all_encodings)
        #
        # similarity_matrix = cosine_similarity(all_encodings)
        #
        # plt.figure(figsize=(12, 10))
        # sns.heatmap(
        #     similarity_matrix,
        #     cmap="Wistia",
        #     annot=False,
        #     vmin=0,
        #     vmax=1
        # )
        #
        # plt.title(f'Inter-image Similarity Heatmap - {directory}')
        # plt.xlabel("Image Index")
        # plt.ylabel("Image Index")
        # plt.tight_layout()
        #
        # # Save the figure
        # plt.savefig(os.path.join(evaluation_result_path, "image_similarity_heatmap.png"), dpi=300)
        # plt.close()

        #----------------------------------------------------------------------------------------------------
        # WRITER CLASSIFICATION
        #----------------------------------------------------------------------------------------------------
        print("Writer Classification: ", directory)
        training_set = []
        training_labels = []
        test_set = []
        test_labels = []

        author_count = defaultdict(int)
        files = sorted(os.listdir(vlad_inference_output_path))
        # random.shuffle(files)
        for file in files:
            if file.endswith(".npy"):
                file_path = os.path.join(vlad_inference_output_path, file)
                encoding = np.load(file_path)
                base_image_author = file.split("_")[0]
                if author_count[base_image_author] < 2:
                    training_set.append(encoding)
                    training_labels.append(base_image_author)
                    author_count[base_image_author] = author_count.get(base_image_author, 0) + 1
                else:
                    test_set.append(encoding)
                    test_labels.append(base_image_author)

        print("Testset:", len(test_set))
        print("Trainingset:", len(training_set))
        training_set = np.vstack(training_set)
        test_set = np.vstack(test_set)

        nn = KNeighborsClassifier(n_neighbors=5)
        nn.fit(training_set, training_labels)
        distances, indices = nn.kneighbors(test_set, n_neighbors=5)
        training_labels = np.array(training_labels)

        top1_count = 0
        top5_count = 0
        for i, neighbor_indices in enumerate(indices):
            top5_predictions = training_labels[neighbor_indices]
            if test_labels[i] == top5_predictions[0]:
                top1_count += 1
                top5_count += 1
            elif test_labels[i] in top5_predictions:
                top5_count += 1

        total = len(indices)
        top1_accuracy = (top1_count / total) * 100
        top5_accuracy = (top5_count / total) * 100

        table_data = [
            ["Mohammed et al. [23]", "26", ""],
            ["Nasir & Siddiqi [24]", "54", ""],
            ["Nasir et al. [25]", "64", ""],
            ["AngU-Net + SIFT + NN", "47", "83"],
            ["AngU-Net + SIFT + SVM", "57", "87"],
            ["AngU-Net + R-SIFT + NN", "53", "77"],
            ["AngU-Net + R-SIFT + SVM", "60", "80"],
            [directory, f"{round(top1_accuracy)}", f"{round(top5_accuracy)}"]
        ]
        headers = ["Method", "Top-1", "Top-5"]
        print(tabulate(table_data, headers, tablefmt="grid"))
        with open(evaluation_result_path + "/Classification.txt", "w") as f:
            f.write(tabulate(table_data, headers, tablefmt="grid"))

        # Generate confusion matrix TODO: REWRITE
        # unique_labels = sorted(set(training_labels + test_labels))
        # cm = confusion_matrix(
        #     test_labels,
        #     predictions,
        #     labels=unique_labels
        # )
        #
        # # Plot the confusion matrix
        # plt.figure(figsize=(12, 10))
        # sns.heatmap(
        #     cm,
        #     annot=True,
        #     fmt='d',
        #     cmap='Blues',
        #     xticklabels=unique_labels,
        #     yticklabels=unique_labels
        # )
        # plt.title(f'Writer Classification Confusion Matrix - {directory}')
        # plt.xlabel('Predicted Label')
        # plt.ylabel('True Label')
        # plt.tight_layout()
        #
        # plt.savefig(os.path.join(evaluation_result_path, "writer_classification_confusion_matrix.png"), dpi=300)
        # plt.close()

if __name__ == "__main__":
    main()
