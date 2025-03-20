import json
import os
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
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.neighbors import NearestNeighbors

WINDOW_SIZE = 32
VLAD_NUM_CLUSTER = 128
RESNET_NUM_CLUSTER = 5000
WHITE_PIXEL_THRESHOLD = 0.85
BLACK_PIXEL_THRESHOLD = 0.6


def get_patch(img, px, py):
    half_win_size = int(WINDOW_SIZE / 2)
    if not (half_win_size < px < img.shape[1] - half_win_size and half_win_size < py < img.shape[0] - half_win_size):
        # if px - half_win_size < 0 or px + half_win_size > img.shape[1] or py - half_win_size < 0 or py + half_win_size > img.shape[0]:
        return None

    roi = img[py - half_win_size:py + half_win_size, px - half_win_size:px + half_win_size]
    assert roi.shape == (half_win_size * 2, half_win_size * 2), 'shape of the roi is not (%d,%d). It is (%d,%d)' % \
                                                                (half_win_size * 2, half_win_size * 2,
                                                                 roi.shape[0], roi.shape[1])
    return roi

def inference_samples_patches_from_image(model_path, image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = resnet.__dict__["resnet20"](num_classes=RESNET_NUM_CLUSTER)
    checkpoint = torch.load(model_path, map_location=device)
    state = checkpoint.get('state_dict', checkpoint)
    fixed_state = OrderedDict((k.replace("module.", "") if k.startswith("module.") else k, v)
                              for k, v in state.items())
    net.load_state_dict(fixed_state)
    net.eval()
    net = net.to(device)

    img = cv2.imread(image_path)
    if img is None:
        return np.zeros((0, VLAD_NUM_CLUSTER))
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

        if extract_patches.black_pixels(patch_gray) > (BLACK_PIXEL_THRESHOLD * num_pix) or extract_patches.white_pixels(patch_gray) > (WHITE_PIXEL_THRESHOLD * num_pix):
            continue

        seen.add((x, y))
        patch_color = cv2.cvtColor(patch_gray, cv2.COLOR_GRAY2BGR)
        patches.append(patch_color)


    feat_list, idx = [], 0
    current_out = None

    def hook(module, inp, out):
        nonlocal current_out
        current_out = out.detach()

    handle = net.layer3.register_forward_hook(hook)
    try:
        with torch.no_grad():
            while idx < len(patches):
                batch = patches[idx:idx + 128]
                batch_tensor = torch.stack([transforms.ToTensor()(p) for p in batch]).to(device)
                net(batch_tensor)
                pooled = torch.nn.functional.avg_pool2d(current_out, current_out.size(3))
                feat_list.append(pooled.view(pooled.size(0), -1).cpu().numpy())
                idx += 128
    finally:
        handle.remove()

    return np.vstack(feat_list) if feat_list else np.zeros((0, VLAD_NUM_CLUSTER))

def main():

    path = "./extract_patches_input"
    dirs = os.listdir(path=path)
    for directory in dirs:

        # ---------------------------------------------------------------------------------------------------
        # EXTRACT PATCHES
        # ---------------------------------------------------------------------------------------------------
        print("Extracting Patches: ", directory)
        input_path = os.path.join(path, directory)
        output_path = os.path.join("extract_patches_output", directory)
        resnet_output_path = os.path.join("resnet20_output", directory)


        # if not os.path.exists("extract_patches_output"):
        #     os.mkdir("extract_patches_output")
        # if not os.path.exists(output_path):
        #     os.mkdir(output_path)
        # else:
        #     shutil.rmtree(output_path)
        #     os.mkdir(output_path)
        #
        #
        # result = subprocess.run([sys.executable, 'extract_patches.py', "--in_dir", input_path, "--out_dir", output_path, "--num_of_clusters", f"{RESNET_NUM_CLUSTER}", "--centered", "True", "--black_pixel_thresh", f"{BLACK_PIXEL_THRESHOLD}", "--white_pixel_thresh", f"{WHITE_PIXEL_THRESHOLD}", "--scale", "1"]
        #                        , stdout=None, stderr=None)
        # print(result.stdout)
        # center_path = os.path.join(output_path, 'centers.pkl')
        # parameter_path = os.path.join(output_path, 'db-creation-parameters.json')
        # if os.path.exists(center_path):
        #     os.remove(center_path)
        # if os.path.exists(parameter_path):
        #     os.remove(parameter_path)

        # ---------------------------------------------------------------------------------------------------
        # TRAIN RESNET
        # ---------------------------------------------------------------------------------------------------
        print("Training Resnet: ", directory)
        if not os.path.exists("resnet20_output"):
            os.mkdir("resnet20_output")
        if not os.path.exists(resnet_output_path):
            os.mkdir(resnet_output_path)


        result = subprocess.run([sys.executable, 'test.py',
            "--arch", "resnet20",
            "--workers", "8",
            "--epochs", "200",
            "--batch-size", "2048",
            "--lr", "0.3",
            "--momentum", "0.9",
            "--weight-decay", "1e-4",
            "--print-freq", "120",
            "--output_dir", resnet_output_path,
            "--save-every", "50",
            "--input_dir", output_path],
            stdout=None, stderr=None)
        print(result.stdout)

        # ---------------------------------------------------------------------------------------------------
        # TRAIN VLAD
        # ---------------------------------------------------------------------------------------------------
        print("Training VLAD: ", directory)

        vlad_inputs = []
        for input_image in os.listdir(path=os.path.join(path, directory)):
            print(f"Processing {input_image}")
            features = inference_samples_patches_from_image(os.path.join(resnet_output_path, "model.th"),
                                             os.path.join(path, directory, input_image))
            vlad_inputs.append(features)

        vlad_inputs = np.vstack(vlad_inputs)
        v = vlad.VLAD(n_clusters=VLAD_NUM_CLUSTER, gmp=True, gamma=800)
        v._train_impl(vlad_inputs)

        vlad_train_output_path = os.path.join("vlad_train_output", directory)

        if not os.path.exists("vlad_train_output"):
            os.makedirs("vlad_train_output")
        if not os.path.exists(vlad_train_output_path):
            os.makedirs(vlad_train_output_path)
        v.save(os.path.join(vlad_train_output_path, "model.pkl"))

        # ---------------------------------------------------------------------------------------------------
        # INFERENCE VLAD
        # ---------------------------------------------------------------------------------------------------
        print("Inferencing VLAD: ", directory)
        vlad_inference_output_path = os.path.join("vlad_inference_output", directory)
        if not os.path.exists("vlad_inference_output"):
            os.makedirs("vlad_inference_output")
        if not os.path.exists(vlad_inference_output_path):
            os.makedirs(vlad_inference_output_path)


        v.load(os.path.join(vlad_train_output_path, "model.pkl"))
        for input_image in os.listdir(path=os.path.join(path, directory)):
            print(f"Inferencing {input_image} with VLAD")

            features = inference_samples_patches_from_image(os.path.join(resnet_output_path, "model.th"),
                                             os.path.join(path, directory, input_image))
            vlad_output = v.encode(features)
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
            for input_image in os.listdir(path=os.path.join(path, directory)):
                to_compare = os.path.join(vlad_inference_output_path, input_image.replace("tiff", "npy"))
                author = input_image.split("_")[0]
                to_compare_encoding = np.load(to_compare)

                stored_encodings = []
                file_paths = []
                for filename in os.listdir(vlad_inference_output_path):
                    if filename.endswith(".npy") and filename != os.path.basename(to_compare):
                        file_path = os.path.join(vlad_inference_output_path, filename)
                        stored_encodings.append(np.load(file_path))
                        file_paths.append(filename)

                stored_encodings = np.vstack(stored_encodings)
                similarities = cosine_similarity(to_compare_encoding.reshape(1, -1), stored_encodings)[0]
                indices = np.argsort(similarities)[::-1][:10]
                most_similar_files = [file_paths[i].split('_')[0] for i in indices]
                stats.append({
                    "author": author,
                    "top1": author in most_similar_files[:1],
                    "top5": author in most_similar_files[:5],
                    "top10": author in most_similar_files[:10]
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
            [directory, f"{top1_accuracy:.2f}", f"{top5_accuracy:.2f}", f"{top10_accuracy:.2f}", ""]
        ]
        headers = ["Method", "Top-1", "Top-5", "Top-10", "mAP"]
        print(tabulate(table_data, headers, tablefmt="grid"))
        with open(evaluation_result_path + "/Retrieval.txt", "w") as f:
            f.write(tabulate(table_data, headers, tablefmt="grid"))

        # ---------------------------------------------------------------------------------------------------
        # HEATMAP GENERATION
        # ---------------------------------------------------------------------------------------------------
        print("Generating Heatmaps: ", directory)


        author_encodings = defaultdict(list)
        for filename in os.listdir(vlad_inference_output_path):
            if filename.endswith(".npy"):
                file_path = os.path.join(vlad_inference_output_path, filename)
                author = filename.split("_")[0]
                encoding = np.load(file_path)
                author_encodings[author].append(encoding)

        authors = sorted(author_encodings.keys())
        similarity_matrix = np.zeros((10,10))

        for i, author1 in enumerate(authors):
            for j, author2 in enumerate(authors):
                encodings_author1 = np.vstack(author_encodings[author1])
                encodings_author2 = np.vstack(author_encodings[author2])
                similarities = cosine_similarity(encodings_author1, encodings_author2)
                avg_similarity = np.mean(similarities)
                similarity_matrix[i, j] = avg_similarity

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            similarity_matrix,
            cmap="Wistia",
            xticklabels=authors,
            yticklabels=authors,
            annot=False,
            vmin=0,
            vmax=1
        )
        plt.title(f'Inter-scribe Similarity Heatmap - {directory}')
        plt.tight_layout()

        plt.savefig(os.path.join(evaluation_result_path, "scribe_similarity_heatmap.png"), dpi=300)
        plt.close()


        all_encodings = []
        for filename in os.listdir(vlad_inference_output_path):
            if filename.endswith(".npy"):
                file_path = os.path.join(vlad_inference_output_path, filename)
                encoding = np.load(file_path)
                all_encodings.append(encoding)


        all_encodings = np.vstack(all_encodings)

        similarity_matrix = cosine_similarity(all_encodings)

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            similarity_matrix,
            cmap="Wistia",
            annot=False,
            vmin=0,
            vmax=1
        )

        plt.title(f'Inter-image Similarity Heatmap - {directory}')
        plt.xlabel("Image Index")
        plt.ylabel("Image Index")
        plt.tight_layout()

        # Save the figure
        plt.savefig(os.path.join(evaluation_result_path, "image_similarity_heatmap.png"), dpi=300)
        plt.close()

        #----------------------------------------------------------------------------------------------------
        # WRITER CLASSIFICATION
        #----------------------------------------------------------------------------------------------------

        training_set = []
        training_labels = []
        test_set = []
        test_labels = []

        author_count = defaultdict(int)
        for file in sorted(os.listdir(vlad_inference_output_path)):
            if file.endswith(".npy"):
                file_path = os.path.join(vlad_inference_output_path, file)
                encoding = np.load(file_path).reshape(-1)
                author = file.split("_")[0]
                if author_count[author] >= 2:
                    test_set.append(encoding)
                    test_labels.append(author)
                else:
                    training_set.append(encoding)
                    training_labels.append(author)
                    author_count[author] = author_count.get(author, 0) + 1

        training_set = np.array(training_set)
        test_set = np.array(test_set)

        # Nearest neighbor implementation
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(training_set)

        # For each test sample, find the nearest neighbor
        distances, indices = nn.kneighbors(test_set)
        predictions = [training_labels[idx[0]] for idx in indices]

        # Calculate top-1 and top-5 accuracy
        top1_correct = sum(1 for pred, true in zip(predictions, test_labels) if pred == true)
        top1_accuracy = (top1_correct / len(test_labels)) * 100 if test_labels else 0

        # For top-5, we need to find 5 nearest neighbors
        nn5 = NearestNeighbors(n_neighbors=5)
        nn5.fit(training_set)
        distances5, indices5 = nn5.kneighbors(test_set)
        top5_correct = 0
        for i, idx_list in enumerate(indices5):
            top5_predictions = [training_labels[idx] for idx in idx_list]
            if test_labels[i] in top5_predictions:
                top5_correct += 1
        top5_accuracy = (top5_correct / len(test_labels)) * 100 if test_labels else 0

        # Generate confusion matrix
        unique_labels = sorted(set(training_labels + test_labels))
        cm = confusion_matrix(
            test_labels,
            predictions,
            labels=unique_labels
        )

        # Plot the confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=unique_labels,
            yticklabels=unique_labels
        )
        plt.title(f'Writer Classification Confusion Matrix - {directory}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()

        plt.savefig(os.path.join(evaluation_result_path, "writer_classification_confusion_matrix.png"), dpi=300)
        plt.close()

        table_data = [
            ["Mohammed et al. [23]", "26", ""],
            ["Nasir & Siddiqi [24]", "54", ""],
            ["Nasir et al. [25]", "64", ""],
            ["AngU-Net + SIFT + NN", "47", "83"],
            ["AngU-Net + SIFT + SVM", "57", "87"],
            ["AngU-Net + R-SIFT + NN", "53", "77"],
            ["AngU-Net + R-SIFT + SVM", "60", "80"],
            [directory, f"{top1_accuracy:.2f}", f"{top5_accuracy:.2f}"]
        ]
        headers = ["Method", "Top-1", "Top-5"]
        print(tabulate(table_data, headers, tablefmt="grid"))
        with open(evaluation_result_path + "/Classification.txt", "w") as f:
            f.write(tabulate(table_data, headers, tablefmt="grid"))

if __name__ == "__main__":
    main()
