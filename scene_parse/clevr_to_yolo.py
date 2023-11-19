import json
import os
import random
import shutil

from flatten_rle import *

# Set the paths to the Clevr mini dataset
data_dir = "scene_parse/CLEVR_mini"
target_dir = "scene_parse/dataset"
images_dir = os.path.join(data_dir, "images")
annotations_file = os.path.join(data_dir, "CLEVR_mini_coco_anns.json")

# Set the percentage of images to use for validation
validation_percent = 0.2

def split() :
    # Load the annotations file
    with open(annotations_file, "r") as f:
        annotations = json.load(f)
        scenes = annotations["scenes"]

    # Get a list of all image filenames
    image_filenames = [os.path.join(images_dir, x["image_filename"]) for x in scenes]

    # Shuffle the image filenames
    random.shuffle(image_filenames)

    # Split the image filenames into train and validation sets
    num_validation = int(len(image_filenames) * validation_percent)
    validation_filenames = image_filenames[:num_validation]
    train_filenames = image_filenames[num_validation:]

    # Create the validation and train directories
    validation_dir = os.path.join(target_dir, "val")
    train_dir = os.path.join(target_dir, "train")
    os.makedirs(validation_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)

    # Copy the validation images and annotations to the validation directory
    for filename in validation_filenames:
        shutil.copy(filename, validation_dir)
        annotation = next(x for x in scenes if x["image_filename"] == os.path.basename(filename))
        with open(os.path.join(validation_dir, f"{os.path.splitext(os.path.basename(filename))[0]}.json"), "w") as f:
            json.dump(annotation, f)

    # Copy the train images and annotations to the train directory
    for filename in train_filenames:
        shutil.copy(filename, train_dir)
        annotation = next(x for x in scenes if x["image_filename"] == os.path.basename(filename))
        with open(os.path.join(train_dir, f"{os.path.splitext(os.path.basename(filename))[0]}.json"), "w") as f:
            json.dump(annotation, f)

def move_in_folder() :

    validation_dir = os.path.join(target_dir, "val")
    train_dir = os.path.join(target_dir, "train")

    # Move the images into the images directory and the annotations into the labels directory
    for filename in os.listdir(validation_dir):
        # Move the image in the images folder (if it ends with .png)
        if filename.endswith(".png"):
            shutil.move(os.path.join(validation_dir, filename), os.path.join(validation_dir, "images", filename))
        # Move the annotation in the labels folder (if it ends with .json)
        if filename.endswith(".json"):
            shutil.move(os.path.join(validation_dir, filename), os.path.join(validation_dir, "labels", filename))
    
    for filename in os.listdir(train_dir):
        # Move the image in the images folder (if it ends with .png)
        if filename.endswith(".png"):
            shutil.move(os.path.join(train_dir, filename), os.path.join(train_dir, "images", filename))
        # Move the annotation in the labels folder (if it ends with .json)
        if filename.endswith(".json"):
            shutil.move(os.path.join(train_dir, filename), os.path.join(train_dir, "labels", filename))

# Create a txt file for each image from the annotation json file
def create_txt(annotation, dict_class, dir) :

    # Create a empty txt file
    f = open(os.path.join(dir, f"{os.path.splitext(os.path.basename(annotation['image_filename']))[0]}.txt"), "w")
    # Loop over the objects in the scene
    for o in annotation["objects"]:
        # Get the class of the object from the annotation : Size_Color_Material_Shape
        name = o["size"].capitalize() + "_" + o["color"].capitalize() + "_" + o["material"].capitalize() + "_" + o["shape"].capitalize()
        # Get the class index from the dictionary
        class_index = dict_class[name]
        # Get the binary mask of the object
        mask = o["mask"]
        # Flatten the mask
        yolo_format = flatten_rle(mask)
        # Write the class index and the mask in the txt file
        # With this format : <class_index> <x1> <y1> <x2> <y2> ... <xn> <yn>
        f.write(f"{class_index} {' '.join(map(str, yolo_format))}\n")
    f.close()

def create_txt_files() :

    # Open the class dictionary in the "dict_classes.json" file
    class_dict = json.load(open(target_dir + "/dict_classes.json"))

    # For the train and validation directories
    dir = os.path.join(target_dir, "train")
    for filename in os.listdir(os.path.join(dir, "labels")):
        if filename.endswith(".txt"):
            continue
        print(filename)
        # Open the annotation json file
        annotation = json.load(open(os.path.join(dir, "labels", filename)))
        # Create the txt file
        create_txt(annotation, class_dict, os.path.join(dir, "labels"))
    
    dir = os.path.join(target_dir, "val")
    for filename in os.listdir(os.path.join(dir, "labels")):
        if filename.endswith(".txt"):
            continue
        print(filename)
        # Open the annotation json file
        annotation = json.load(open(os.path.join(dir, "labels", filename)))
        # Create the txt file
        create_txt(annotation, class_dict, os.path.join(dir, "labels"))

# Remove json files
def remove_json() :

    # For the train directories
    dir = os.path.join(target_dir, "train")
    for filename in os.listdir(os.path.join(dir, "labels")):
        if filename.endswith(".json"):
            os.remove(os.path.join(dir, "labels", filename))
    
    dir = os.path.join(target_dir, "val")
    for filename in os.listdir(os.path.join(dir, "labels")):
        if filename.endswith(".json"):
            os.remove(os.path.join(dir, "labels", filename))

# Remove all files in a directory 
def empty_all() :
    
        # For the train directories
        dir = os.path.join(target_dir, "train")
        for filename in os.listdir(os.path.join(dir, "labels")):
            os.remove(os.path.join(dir, "labels", filename))
        for filename in os.listdir(os.path.join(dir, "images")):
            if filename.endswith(".png"):
                os.remove(os.path.join(dir, "images", filename))
        
        dir = os.path.join(target_dir, "val")
        for filename in os.listdir(os.path.join(dir, "labels")):
            os.remove(os.path.join(dir, "labels", filename))
        for filename in os.listdir(os.path.join(dir, "images")):
            if filename.endswith(".png"):
                os.remove(os.path.join(dir, "images", filename))

def set_split() :

    # List images in val folder
    dir = os.path.join(target_dir, "val")
    list_val = os.listdir(os.path.join(dir, "images"))

    # Load the annotations file
    with open(annotations_file, "r") as f:
        annotations = json.load(f)
        scenes = annotations["scenes"]
        for s in scenes :
            if s["image_filename"] in list_val :
                s["split"] = "val"
            else :
                s["split"] = "train"

    # Save the annotations file
    with open(annotations_file, "w") as f:
        json.dump(annotations, f)


#TODO Run the complete pipeline (copy on run_test.py to do so) => Extraire les résultats
#TODO Bonus : Faire la partie CE

if __name__ == "__main__":
    
    empty_all()
    split()
    move_in_folder()
    create_txt_files()
    remove_json()
    set_split()
    