"""Preprocess all inputs and store as individual files in folders
"""
import os
from conversion import (FloorplanRaw, FloorplanSVG, read_room2class_condensed,
                        get_class2color)
import numpy as np

if __name__ == "__main__":
    # Mappings for dataset
    room2class = read_room2class_condensed("room_types.txt")
    class2room = {v: k for k, v in room2class.items()}
    class2color = get_class2color(max(room2class.values()))

    # Process all the data
    dataset_dir = "./data/cubicasa5k"
    txts = ["train.txt", "val.txt", "test.txt"]
    for txt in txts:
        dataset_txt = dataset_dir + "/" + txt
        with open(dataset_txt, "r") as f:
            line = f.readline().strip()
            i = 0
            while line:
                i += 1
                print(f"process {dataset_txt} line: {i}")

                data_folder = dataset_dir + line
                raw = FloorplanRaw(data_folder)
                svg = FloorplanSVG(data_folder, raw.original_shape, room2class)

                # Create folder to store data
                sample_dir = dataset_dir + "/" + txt.split(".")[0] + line
                if not os.path.exists(sample_dir):
                    os.makedirs(sample_dir)

                # Store input images and semantic map labels
                input_npz = sample_dir + "input.npy"
                np.save(open(input_npz, "wb"), raw.image)

                label_npz = sample_dir + "label.npy"
                np.save(open(label_npz, "wb"), svg.semantic_map)

                line = f.readline().strip()
