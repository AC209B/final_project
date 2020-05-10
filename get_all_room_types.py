"""Process the data folder to assemble all the possible room types and store as a txt file
"""
from os import walk
from lxml import etree
from conversion import FloorplanSVG


if __name__ == "__main__":
    # Parse image directories
    datadir = "./data/cubicasa5k/"
    categories = ["colorful", "high_quality", "high_quality_architectural"]
    image_dirs = {cat: None for cat in categories} # list of dir names in each category
    for cat in categories:
        path = datadir + cat
        for _, dirnames, _ in walk(path):
            image_dirs[cat] = dirnames
            break

    # Get all rooms types in dataset
    room_types = set()

    for cat, dirs in image_dirs.items():
        for dir_ in dirs:
            svg_path = datadir + cat + "/" + dir_ + "/model.svg"

            # Parse room types in file
            tree = etree.parse(open(svg_path, 'r'))
            for element in tree.iter():
                if FloorplanSVG.get_tag(element) == "g" and FloorplanSVG.is_room(element):
                    room_type = FloorplanSVG.get_room_type(element)
                    room_types.add(room_type)


    # Save room types into a file as name to id correspondence
    room_types = list(room_types)
    fname = "room_types.txt"
    with open(fname, "w") as f:
        for idx, room_type in enumerate(room_types):
            f.write(f"{room_type}: {idx+1}\n")
