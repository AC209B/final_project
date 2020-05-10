"""Preprocess all inputs as store as npy files
"""
import os
from conversion import (FloorplanRaw, FloorplanSVG, read_room2class_condensed,
                        get_class2color)
import numpy as np

if __name__ == "__main__":
    # Mappings for dataset
    room2class = read_room2class_condensed("room_types_succinct.txt")
    class2room = {v: k for k, v in room2class.items()}
    class2color = get_class2color(max(room2class.values()))

    dataset_dir = "./data/cubicasa5k"

    txts = ["train.txt", "val.txt", "test.txt"]

    for txt in txts:
        dataset_txt = dataset_dir + "/" + txt
        inputs = []
        labels = []
        with open(dataset_txt, "r") as f:
            line = f.readline().strip()
            i = 0
            while line:
                i += 1
                print(f"process {dataset_txt} line: {i}")

                data_folder = dataset_dir + line
                raw = FloorplanRaw(data_folder)
                svg = FloorplanSVG(data_folder, raw.original_shape, room2class)

                inputs.append(raw.image)
                labels.append(svg.semantic_map)

                line = f.readline().strip()

        # Store input arrays
        inputs = np.array(inputs)
        inputs_npz = dataset_dir + "/" + txt.split(".")[0] + "_inputs.npy"
        np.save(open(inputs_npz, "wb"), inputs)

        # Store labels arrays
        labels = np.array(labels)
        labels_npz = dataset_dir + "/" + txt.split(".")[0] + "_labels.npy"
        np.save(open(labels_npz, "wb"), labels)
