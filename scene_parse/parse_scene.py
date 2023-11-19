# Script that will be used to parse a scene data
from ultralytics import YOLO
import torch
import os
import sys
import random
import json
import numpy as np

# Set the current directory to the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene_parse.coords_projector import ObjectLocalizationMLP


path = "runs/segment/train1/weights"

val_images = "scene_parse/train_dataset/val/images"
class_dict = "scene_parse/train_dataset/dict_classes.json"

class SceneParser() : 

    def init(self, name : str = "best") :
        self.model = YOLO(path + "/" + name + ".pt")
        self.class_dict = json.load(open(class_dict))
        self.attr_model = ObjectLocalizationMLP(6, [8,16,8], 2)
        self.attr_model.load_state_dict(torch.load("scene_parse/attr_model.pth"))

    def predict(self, path_to_image : str) :
        return self.model(path_to_image, verbose=False)
    
    def ramdom_val_image(self) -> str :
        name = random.choice(os.listdir(val_images))
        return os.path.join(val_images, random.choice(os.listdir(val_images)))
    
    def get_attributes(self, result : list) -> list[float, float]:
        """
        Return the attributes of the objects in the image (coordinates, size, etc.)
        
        Parameters
        ----------
        result : list
            The result of the prediction of the model
            
        Returns
        -------
        list
            The attributes of the objects in the image
        """
        
        attributes = []
        for r in result :
            boxes = r.boxes
            for b in boxes :
                t = b.xyxy.tolist()[0]
                # Get center of the box
                x = (t[0] + t[2]) / 2
                y = (t[1] + t[3]) / 2

                attr = self.get_attr_from_cls(b.cls.tolist()[0])

                size = 0 if attr[0] == "Small" else 1
                # One hot encoding of the shape (cube, sphere, or cylinder)
                shape = [1, 0, 0] if attr[3] == "cube" else [0, 1, 0] if attr[3] == "sphere" else [0, 0, 1]
                coords_3d = self.attr_model(torch.tensor(shape + [size, x, y], dtype=torch.float32))
                coords_3d = coords_3d.tolist()

                z = 0.35
                if size==1 :
                    z += 0.35
                attributes.append(attr + [round(coords_3d[0], 2), round(coords_3d[1], 2), z])

        return attributes
    
    def get_attr_from_cls(self, cls : int) -> list[str, str, str, str] :
        """
        Finds the attributes of the object from its class

        Parameters
        ----------
        cls : int
            The class of the object
        
        Returns
        -------
        list
            The attributes of the object (size, color, material, shape)
        """

        # Find in the dict which key has the value cls
        for key, value in self.class_dict.items() :
            if value == cls :
                return key.split("_")
    
    def demo(self) :
        path_to_image = self.ramdom_val_image()
        result = self.predict(path_to_image)
        print(self.get_attributes(result))


def build_dict(attributes : list, index : str) -> dict :
    """
    Build the dictionary that will be used to create the json file

    Parameters
    ----------
    attributes : list
        The attributes of the objects in the image
    index : str
        The index of the image
    
    Returns
    -------
    dict
        The dictionary that will be used to create the json file
    """

    RIGHT = [0.6563112735748291, 0.7544902563095093, -0.0]
    FRONT = [0.754490315914154, -0.6563112735748291, -0.0]

    # Create the dictionary
    d = {
        "image_index": index,
        "objects": []
    }

    # Loop over the attributes
    for attr in attributes :
        # Create the object dictionary
        coords_3d = [attr[4], attr[5], attr[6]]
        position = [np.dot(coords_3d, RIGHT),
                    np.dot(coords_3d, FRONT),
                    coords_3d[-1]]

        o = {
            "position": position,
            "color": attr[1].lower(),
            "material": attr[2].lower(),
            "shape": attr[3].lower(),
            "size": attr[0].lower()
        }
        # Append the object dictionary to the objects list
        d["objects"].append(o)
    
    return d

def write_json(parser : SceneParser, path : str = "scene_parse", split : str = "val") :

    """
    Create the json file with the abstract attributes of the objects in the scene
    """
    images_dir = path + "/images/" + split

    res_dict = {"scenes" : []}

    json_file = "scene_parse/" + split + "_scene.json"

    file_list = os.listdir(images_dir)
    file_list.sort()

    for img_filename in file_list:
        index = int(img_filename.split("_")[-1].split(".")[0])

        if len(res_dict["scenes"])%10 == 0 :
            print(str(len(res_dict["scenes"])//10)+"%")

        path_to_img = os.path.join(images_dir, img_filename)
        result = parser.predict(path_to_img)

        res_dict["scenes"].append(build_dict(parser.get_attributes(result), index))
    
    # Order scenes by image_index
    
    # Save the json
    with open(json_file, "x") as f:
        json.dump(res_dict, f)


if __name__ == "__main__" :
    parser = SceneParser()
    parser.init()
    
    write_json(parser, split="test")
    


    
        
    