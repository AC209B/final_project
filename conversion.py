"""Classes for storing the raw inputs and svgs in structured ways
"""
import matplotlib.pyplot as plt
from skimage.transform import resize
import skimage
import numpy as np
import cairosvg
from PIL import Image
import io
from lxml import etree

from constants import INPUT_SHAPE, OUTPUT_SHAPE


class FloorplanRaw:
    """Class for raw floorplan image
    """
    def __init__(self, folder_path):
        fpath = folder_path + "F1_scaled.png"
        image =  plt.imread(fpath)
        self.original_shape = image.shape[:2]
        self.image = resize(image, INPUT_SHAPE)

    def show(self):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.image)


def svgRead(filename):
   """Load an SVG file and return image in Numpy array"""
   # Make memory buffer
   mem = io.BytesIO()
   # Convert SVG to PNG in memory
   cairosvg.svg2png(url=filename, write_to=mem)
   # Convert PNG to Numpy array
   return np.array(Image.open(mem))


class Polygon:
    def __init__(self, polygon_element, scaling_x, scaling_y):
        points_str = polygon_element.get("points").split(" ")
        points = []
        for s in points_str:
            if "," in s:
                a, b = s.split(",")
                points.append([int(float(a)*scaling_x),
                               int(float(b)*scaling_y)])
        self.points = points

    @property
    def x(self):
        return [p[0] for p in self.points]

    @property
    def y(self):
        return [p[1] for p in self.points]


def read_room2class(fpath):
    room2class = {}
    with open(fpath, 'r') as f:
        line = f.readline()
        while line:
            room_type, idx = line.split(":")
            room2class[room_type] = int(idx)
            line = f.readline()

    return room2class


def read_room2class_condensed(fpath):
    room2class = {}
    with open(fpath, 'r') as f:
        line = f.readline()
        while line:
            room_types, idx = line.split(":")
            room2class[room_types] = int(idx)
            line = f.readline()

    return room2class


def get_class2color(n, name='hsv'):
    cmap = plt.cm.get_cmap(name, n)

    class2color = {}
    for idx in range(1, n):
        class2color[idx] = cmap(idx)
    return class2color


class FloorplanSVG:
    """Class for floorplan SVG file
    """
    def __init__(self, folder_path, original_shape, room2class):
        self.original_shape = original_shape

        scaling_x = OUTPUT_SHAPE[1]/original_shape[1]
        scaling_y = OUTPUT_SHAPE[0]/original_shape[0]

        # Extract rooms as polygons
        rooms = []
        fpath = folder_path + "model.svg"
        tree = etree.parse(open(fpath, 'r'))
        for element in tree.iter():
            if self.get_tag(element) == "g" and self.is_room(element):
                room_type = self.get_room_type(element)
                polygon = Polygon(element.getchildren()[0],
                                  scaling_x, scaling_y)
                rooms.append([room_type, polygon])

        # Create 2D array where each value is class id
        semantic_map = np.zeros(dtype=np.int8, shape=OUTPUT_SHAPE)
        for room_type, polygon in rooms:
            rr, cc = skimage.draw.polygon(polygon.y, polygon.x)
            rr = np.clip(rr, 0, OUTPUT_SHAPE[0]-1)
            cc = np.clip(cc, 0, OUTPUT_SHAPE[1]-1)
            for k in room2class:
                if room_type in k.split(","):
                    semantic_map[rr, cc] = room2class[k]
        self.semantic_map = semantic_map

        # image = svgRead(fpath)
        # self.image = resize(image, (256, 256))

    def show(self, base_image, class2color, class2room):
        """Plot the class labelling by color for visualization
        """
        self.show_map(self.semantic_map, base_image, class2color, class2room)

    @staticmethod
    def show_map(semantic_map, base_image, class2color, class2room):
        plt.figure(figsize=(10, 10))
        # color_image = np.zeros(shape=(*self.semantic_map.shape, 3))
        plt.imshow(base_image)
        for cl in class2color:
            coords = np.where(semantic_map == cl)
            if len(coords[0]) == 0: continue

            label = "Others"
            class2label = {
                0: "Empty",
                1: "Entry",
                2: "Outdoor",
                3: "Bathroom",
                4: "Kitchen",
                5: "Livingroom",
                6: "Bedroom",
                7: "Others"
            }
            plt.scatter(coords[1], coords[0], color=class2color[cl], alpha=0.1, label=class2label[cl])
        plt.legend()

    @staticmethod
    def get_tag(element):
        return element.tag.split("}")[1]

    @staticmethod
    def is_room(element):
        cl = element.get("class")
        return cl is not None and cl[:6] == "Space "

    @staticmethod
    def get_room_type(element):
        return element.get("class")[6:]

