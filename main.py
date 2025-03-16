import json
import os
import shutil
import subprocess
import sys

import numpy as np

import vlad
import resnet
import torchvision.transforms as transforms
import torch
from PIL import Image
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors

def inference_samples_patches_from_image(model_path, image_path, max_patches=200):
    # 1. Initialize model with correct architecture and class count
    model = resnet.__dict__["resnet20"](num_classes=5000)

    # 2. Load trained weights with module prefix handling
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    # 3. Prepare image transformation
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # 4. Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    width, height = image.size

    # 5. Calculate all possible patch positions
    patch_size = 32
    stride = 16

    all_positions = []
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            all_positions.append((x, y))

    # 6. Randomly sample patch positions if there are too many
    if len(all_positions) > max_patches:
        import random
        sampled_positions = random.sample(all_positions, max_patches)
    else:
        sampled_positions = all_positions

    # 7. Extract and process sampled patches
    all_features = []
    for x, y in sampled_positions:
        # Extract patch
        patch = image.crop((x, y, x + patch_size, y + patch_size))

        # Transform patch
        patch_tensor = transform(patch).unsqueeze(0)

        # Extract features using hook
        features = None

        def hook_fn(module, input, output):
            nonlocal features
            features = output.detach()

        # Register hook to the layer before linear
        hook = model.layer3.register_forward_hook(hook_fn)

        # Forward pass
        with torch.no_grad():
            model(patch_tensor)

        # Remove hook
        hook.remove()

        # Process features
        pooled_features = torch.nn.functional.avg_pool2d(features, features.size()[3])
        feature_vector = pooled_features.view(pooled_features.size(0), -1)

        # Add to collection
        all_features.append(feature_vector.cpu().numpy())

    # 8. Stack all features into a single array
    if all_features:
        feature_array = np.vstack(all_features)
        return feature_array
    else:
        # Return empty array with correct dimensions if no patches were extracted
        return np.zeros((0, 64))  # Assuming 64-dimensional features


def inference_whole_image(model_path, image_path):
    # 1. Initialize model with correct architecture and class count
    model = resnet.__dict__["resnet20"](num_classes=5000)

    # 2. Load trained weights with module prefix handling
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    # 3. Prepare image transformation
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # 4. Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    all_features = []
    patch_size = 32
    stride = 16

    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            # Extract patch
            patch = image.crop((x, y, x + patch_size, y + patch_size))

            # Transform patch
            patch_tensor = transform(patch).unsqueeze(0)

            # Extract features using hook
            features = None

            def hook_fn(module, input, output):
                nonlocal features
                features = output.detach()

            # Register hook to the layer before linear
            hook = model.layer3.register_forward_hook(hook_fn)

            # Forward pass
            with torch.no_grad():
                model(patch_tensor)

            # Remove hook
            hook.remove()

            # Process features
            pooled_features = torch.nn.functional.avg_pool2d(features, features.size()[3])
            feature_vector = pooled_features.view(pooled_features.size(0), -1)

            # Add to collection
            all_features.append(feature_vector.cpu().numpy())

    # 6. Stack all features into a single array
    if all_features:
        feature_array = np.vstack(all_features)
        return feature_array
    else:
        # Return empty array with correct dimensions if no patches were extracted
        return np.zeros((0, 64))  # Assuming 64-dimensional features

def main():

    path = "./extract_patches_input"
    dirs = os.listdir(path=path)
    for directory in dirs:

        # ---------------------------------------------------------------------------------------------------
        # EXTRACT PATCHES
        # ---------------------------------------------------------------------------------------------------

        # output_path = os.path.join("extract_patches_output", directory)
        # if not os.path.exists("extract_patches_output"):
        #     os.mkdir("extract_patches_output")
        # if not os.path.exists(output_path):
        #     os.mkdir(output_path)
        # else:
        #     shutil.rmtree(output_path)
        #     os.mkdir(output_path)
        # input_path = os.path.join(path, directory)
        # print("Now processing: ", input_path)
        # result = subprocess.run([sys.executable, 'extract_patches.py', "--in_dir", input_path, "--out_dir", output_path, "--num_of_clusters", "5000", "--centered", "True", "--black_pixel_thresh", "0.8", "--white_pixel_thresh", "0.8", "--scale", "1.2"], capture_output=True, text = True)
        # print("Standard Output:")
        # print(result.stdout)
        # print("Standard Error:")
        # print(result.stderr)

        # center_path = os.path.join(output_path, 'centers.pkl')
        # parameter_path = os.path.join(output_path, 'db-creation-parameters.json')
        # if os.path.exists(center_path):
        #     os.remove(center_path)
        # if os.path.exists(parameter_path):
        #     os.remove(parameter_path)

        resnet_output_path = os.path.join("resnet20_output", directory)

        # if not os.path.exists("resnet20_output"):
        #     os.mkdir("resnet20_output")
        # if not os.path.exists(resnet_output_path):
        #     os.mkdir(resnet_output_path)
        # else:
        #     shutil.rmtree(resnet_output_path)
        #     os.mkdir(resnet_output_path)
        #


        # ---------------------------------------------------------------------------------------------------
        # TRAIN RESNET
        # ---------------------------------------------------------------------------------------------------

        # result = subprocess.run([sys.executable, 'test.py',
        #     "--arch", "resnet20",
        #     "--workers", "8",
        #     "--epochs", "200",
        #     "--start-epoch", "0",
        #     "--batch-size", "2048",
        #     "--lr", "0.1",
        #     "--momentum", "0.9",
        #     "--weight-decay", "1e-4",
        #     "--print-freq", "50",
        #     "--output_dir", resnet_output_path,
        #     "--save-every", "10",
        #     "--input_dir", output_path],
        #     stdout=None, stderr=None)
        #
        # print("Standard Output:")
        # print(result.stdout)
        # print("Standard Error:")
        # print(result.stderr)


        # ---------------------------------------------------------------------------------------------------
        # TRAIN VLAD
        # ---------------------------------------------------------------------------------------------------

        # vlad_inputs = []
        # for input_image in os.listdir(path=os.path.join(path, directory)):
        #     print(f"Processing {input_image}")
        #     features = inference_samples_patches_from_image(os.path.join(resnet_output_path, "model.th"),
        #                                      os.path.join(path, directory, input_image))
        #     vlad_inputs.append(features)
        #
        # vlad_inputs = np.vstack(vlad_inputs)
        # v = vlad.VLAD(n_clusters=5000, gmp=True)
        # v._train_impl(vlad_inputs)
        #
        # vlad_train_output_path = os.path.join("vlad_train_output", directory)
        #
        # if not os.path.exists("vlad_train_output"):
        #     os.makedirs("vlad_train_output")
        # if not os.path.exists(vlad_train_output_path):
        #     os.makedirs(vlad_train_output_path)
        # v.save(os.path.join(vlad_train_output_path, "model.pkl"))

        # ---------------------------------------------------------------------------------------------------
        # INFERENCE VLAD
        # ---------------------------------------------------------------------------------------------------

        vlad_inference_output_path = os.path.join("vlad_inference_output", directory)
        # if not os.path.exists("vlad_inference_output"):
        #     os.makedirs("vlad_inference_output")
        # if not os.path.exists(vlad_inference_output_path):
        #     os.makedirs(vlad_inference_output_path)
        #
        # v.load(os.path.join(vlad_train_output_path, "model.pkl"))
        #
        # for input_image in os.listdir(path=os.path.join(path, directory)):
        #     print(f"Inferencing {input_image} with VLAD")
        #
        #     features = inference_samples_patches_from_image(os.path.join(resnet_output_path, "model.th"),
        #                                      os.path.join(path, directory, input_image), max_patches=500)
        #     vlad_output = v.encode(features)
        #     np.save(os.path.join(vlad_inference_output_path, input_image.replace("tiff", "") + "npy"), vlad_output)

        # ---------------------------------------------------------------------------------------------------
        # SIMILARITY CALCULATION
        # ---------------------------------------------------------------------------------------------------

        # similarity_output_path = os.path.join("similarity_output", directory)
        # if not os.path.exists("similarity_output"):
        #     os.makedirs("similarity_output")
        # if not os.path.exists(similarity_output_path):
        #     os.makedirs(similarity_output_path)
        #
        # with open(os.path.join(similarity_output_path, "results.txt"), "w") as f:
        #     f.write("Results for directory: " + directory + "\n")
        #
        #     stats = []
        #     for input_image in os.listdir(path=os.path.join(path, directory)):
        #
        #         to_compare = os.path.join(vlad_inference_output_path, input_image.replace("tiff","npy"))
        #         author = input_image.split("_")[0]
        #         to_compare_encoding = np.load(to_compare)
        #
        #         stored_encodings = []
        #         file_paths = []
        #         for filename in os.listdir(vlad_inference_output_path):
        #             if filename.endswith(".npy") and filename != os.path.basename(to_compare):
        #                 file_path = os.path.join(vlad_inference_output_path, filename)
        #                 stored_encodings.append(np.load(file_path).reshape(1, -1))
        #                 file_paths.append(filename)
        #
        #         stored_encodings = np.vstack(stored_encodings)
        #         nbrs = NearestNeighbors(n_neighbors=10, algorithm='auto').fit(stored_encodings)
        #         distances, indices = nbrs.kneighbors(to_compare_encoding)
        #         most_similar_files = [file_paths[i].split('_')[0] for i in indices.flatten()]
        #
        #         stats.append({"author": author, "top1": author in most_similar_files[:1], "top5": author in most_similar_files[:5], "top10": author in most_similar_files[:10]})
        #         print(most_similar_files)
        #     f.write(json.dumps(stats, indent=4))

        # ---------------------------------------------------------------------------------------------------
        # CALCULATE

        file_path = './similarity_output/grk_without_2019_dplinknet_fmeasure/results.txt'
        with open(file_path, 'r') as file:
            data = json.load(file)

        results = data[1:]  # Skip the first element which is the directory name
        total = len(results)
        top1_count = sum(1 for result in results if result["top1"])
        top5_count = sum(1 for result in results if result["top5"])
        top10_count = sum(1 for result in results if result["top10"])


        top1_accuracy = (top1_count / total) * 100
        top5_accuracy = (top5_count / total) * 100
        top10_accuracy = (top10_count / total) * 100
        mean_accuracy = (top1_accuracy + top5_accuracy + top10_accuracy) / 3

        print(f"Top 1 Accuracy: {top1_accuracy:.2f}%")
        print(f"Top 5 Accuracy: {top5_accuracy:.2f}%")
        print(f"Top 10 Accuracy: {top10_accuracy:.2f}%")
        print(f"mAP: {mean_accuracy / 100:.2f}")


if not os.path.exists("extract_patches_output"):
            os.mkdir("extract_patches_output")





if __name__ == "__main__":
    main()
