import numpy as np
from PIL import Image
import cv2
from shapely.geometry import Polygon
from pycocotools import mask as cocomask

def rle_to_mask(rle):
    # Decode using pycocotools
    return cocomask.decode(rle)

def mask_to_yolov8(mask, image_size):
    # Convert binary mask to YOLOv8 format (flatten polygon)
    polygons = find_contours(mask)
    yolo_format = []

    # Flatten polygon : <x1> <y1> <x2> <y2> ... <xn> <yn>
    for poly in polygons:
        normalized_poly = [coord / image_size[i % 2] for i, coord in enumerate(poly)]
        # If any coordinate is outside the image, print a warning
        if any([coord < 0 or coord > 1 for coord in normalized_poly]):
            print("WARNING: Coordinates outside image")
        yolo_format.extend(normalized_poly)

    return yolo_format

def find_contours(mask) -> np.ndarray:
    res = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = res[0][0]
    contours = contours.reshape((contours.shape[0], contours.shape[2]))
    return contours

# Function to visualize mask (on a black background and with a random color)
def visualize_mask(yolo_format, image_size):
    mask = np.zeros(image_size)
    image_size = [image_size[1], image_size[0]]
    polygons = np.array(yolo_format).reshape((-1, 2))
    polygons = (polygons * np.array(image_size)).astype(np.int32)
    cv2.fillPoly(mask, [polygons], 255)
    mask = mask.astype(np.uint8)
    mask = Image.fromarray(mask)
    mask.show()

def flatten_rle(rle_mask):
    binary_mask = rle_to_mask(rle_mask)
    image_size = rle_mask["size"]
    image_size = [image_size[1], image_size[0]]
    return mask_to_yolov8(binary_mask, image_size)



# Example usage
rle_mask = {"mask": {"size": [320, 480], "counts": "lVb3`0X99G:E:G9G9G8L5M3N1O2O0O2O0O2O001N10001O0O101O00001O000O10001O0000000000001O0000000001O00000000000000001N100000001O0O10001N10000O2N100O2N1O1O2L3L5I6K6J5K6J6J5K6J6J6J7Hhb9"}}
image_size = rle_mask["mask"]["size"]
visualize_mask(flatten_rle(rle_mask["mask"]), image_size)


