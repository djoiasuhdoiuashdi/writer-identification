import os
import shutil
import subprocess
import sys


def main():

    path = "./extract_patches_input"
    dirs = os.listdir(path=path)
    for directory in dirs:
        output_path = os.path.join("extract_patches_output", directory)
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

        center_path = os.path.join(output_path, 'centers.pkl')
        parameter_path = os.path.join(output_path, 'db-creation-parameters.json')
        if os.path.exists(center_path):
            os.remove(center_path)
        if os.path.exists(parameter_path):
            os.remove(parameter_path)


        resnet_output_path = os.path.join("resnet20_output", directory)

        if not os.path.exists("resnet20_output"):
            os.mkdir("resnet20_output")
        if not os.path.exists(resnet_output_path):
            os.mkdir(resnet_output_path)
        else:
            shutil.rmtree(resnet_output_path)
            os.mkdir(resnet_output_path)

        result = subprocess.run([sys.executable, 'train_resnet20.py',
                                                         "--arch", "resnet20",
                                                         "--workers", "4",
                                                         "--epochs", "200",
                                                         "--start-epoch", "0",
                                                         "--batch-size", "128",
                                                         "--lr", "0.1",
                                                         "--momentum", "0.9",
                                                         "--weight-decay", "1e-4",
                                                         "--print-freq", "50",
                                                         "--save-dir", "save_temp",
                                                         "--save-every", "10",
                                 "--input_dir", output_path,
                                 "--output_dir", resnet_output_path],
                                                         capture_output=True, text=True)
        print("Standard Output:")
        print(result.stdout)
        print("Standard Error:")
        print(result.stderr)





if __name__ == "__main__":
    main()
