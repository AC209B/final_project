"""Classes for storing the raw inputs and svgs in structured ways
"""
import matplotlib.pyplot as plt
from skimage.transform import resize
import skimage
import numpy as np
from PIL import Image
from lxml import etree

from constants import INPUT_SHAPE, OUTPUT_SHAPE, class2label


class FloorplanRaw:
    """Class for raw floorplan image
    """
    def __init__(self, folder_path):
        """Read input image in folder, and scale to required input shape
        Args:
            folder_path: the directory path to read input image from
        """
        fpath = folder_path + "F1_scaled.png"
        image =  plt.imread(fpath)
        self.original_shape = image.shape[:2]
        self.image = resize(image, INPUT_SHAPE)

    def show(self):
        """Show the image for visualization
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(self.image)


class Polygon:
    """Class for representing a polygon in an image
    """
    def __init__(self, polygon_element, scaling_x, scaling_y):
        """Read polygon from svg <polygon> tag, and scale properly
        Args:
            polygon_element: svg <polygon> element
            scaling_x: scale factor in x
            scaling_y: scale factor in y
        """
        points_str = polygon_element.get("points").split(" ")
        points = []
        for s in points_str:
            if "," in s:
                a, b = s.split(",")
                points.append([int(float(a)*scaling_x),
                               int(float(b)*scaling_y)])
        # polygon vertices
        self.points = points

    @property
    def x(self):
        """Get all the x coordinates of the vertices of the polygon
        """
        return [p[0] for p in self.points]

    @property
    def y(self):
        """Get all the y coordinates of the vertices of the polygon
        """
        return [p[1] for p in self.points]


def read_room2class_condensed(fpath):
    """Build a room types to class label dictionary
    Args:
        fpath: the file path to the txt file containing {room types: labels}
    """
    room2class = {}
    with open(fpath, 'r') as f:
        line = f.readline()
        while line:
            room_types, idx = line.split(":")
            room2class[room_types] = int(idx)
            line = f.readline()

    return room2class


def get_class2color(n, name='hsv'):
    """Build a class label to RGB dictionary for convenience of visualization
    Args:
        n: number of classes
        name: color space parametrization
    """
    cmap = plt.cm.get_cmap(name, n)

    class2color = {}
    for idx in range(1, n):
        class2color[idx] = cmap(idx)
    return class2color


class FloorplanSVG:
    """Class for floorplan SVG file
    """
    def __init__(self, folder_path, original_shape, room2class):
        """Read the room type semantic map from a svg file
        Args:
            folder_path: the path to the folder containing the svg file
            original_shape: the shape of the original input image
            room2class: mapping from room types to class labels
        """
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

        # Create 2D array where each value is class label
        semantic_map = np.zeros(dtype=np.int8, shape=OUTPUT_SHAPE)
        for room_type, polygon in rooms:
            rr, cc = skimage.draw.polygon(polygon.y, polygon.x)
            # Clip values that go outside the shape of image
            rr = np.clip(rr, 0, OUTPUT_SHAPE[0]-1)
            cc = np.clip(cc, 0, OUTPUT_SHAPE[1]-1)
            for k in room2class:
                if room_type in k.split(","):
                    semantic_map[rr, cc] = room2class[k]
        self.semantic_map = semantic_map

    def show(self, base_image, class2color, class2room):
        """Plot the semantic map by color for visualization
        Args:
            base_image: the image to be overlayed upon
            class2color: class label to RGB mapping
            class2room: class label to room type mapping
        """
        self.show_map(self.semantic_map, base_image, class2color, class2room)

    @staticmethod
    def show_map(semantic_map, base_image, class2color, class2room, ax=None):
        """Same as for `self.show` except semantic map is also provided as an argument
        """
        if ax is None:
            plt.figure(figsize=(10, 10))
            plt.imshow(base_image)
            for cl in class2color:
                coords = np.where(semantic_map == cl)
                if len(coords[0]) == 0: continue

                plt.scatter(coords[1], coords[0], color=class2color[cl], alpha=0.1, label=class2label[cl])
            plt.legend()
        else:
            ax.imshow(base_image)
            for cl in class2color:
                coords = np.where(semantic_map == cl)
                if len(coords[0]) == 0: continue

                ax.scatter(coords[1], coords[0], color=class2color[cl], alpha=0.1, label=class2label[cl])
                ax.axis("off")
            ax.legend()

    @staticmethod
    def get_tag(element):
        """Helper function for getting tag string of a svg element
        Args:
            element: svg element
        """
        return element.tag.split("}")[1]

    @staticmethod
    def is_room(element):
        """Helper function for checking if a svg element is a room
        Args:
            element: svg element
        """
        cl = element.get("class")
        return cl is not None and cl[:6] == "Space "

    @staticmethod
    def get_room_type(element):
        """Helper function for getting the room type string of a svg element
        Args:
            element: svg element
        """
        return element.get("class")[6:]

