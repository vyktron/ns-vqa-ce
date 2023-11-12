from ultralytics import YOLO
import sys

if len(sys.argv) < 2 :
    resume = False
else :
    resume = True
    train_dir = sys.argv[1]
# If no argument is passed, the model will be trained from scratch
if not resume :
    # Load a model
    model = YOLO('yolov8n-seg.yaml')  # build a new model from YAML
    model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)
    model = YOLO('yolov8n-seg.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
else : 
    dir = "runs/segment/" + train_dir
    # Load a model
    model = YOLO(dir + '/weights/last.pt')


train_dir = "scene_parse/dataset"

# Train the model
results = model.train(data=train_dir + '/data.yaml', epochs=60, imgsz=[320,480], batch=16, rect=True, resume=resume)
