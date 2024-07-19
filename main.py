import os
import cv2
import argparse
import numpy as np
import supervision as sv
from ultralytics import YOLO

Description = """
    This tool helps you annotate your data faster. 
    Before using it, you need a lightweight YOLO model trained on your tiny dataset.

    This tool requires three arguments:
    
    1. Model path (--Model)
    2. Images path (--Input)
    3. Output path (--OutPut)

    When you provide these three arguments to the tool, 
    it will automatically display a window that annotates your image. You can choose to save the annotated image or not.

    FOR EXAMPLE:

        In Linux:  
        !python3 main.py --Model "YOUR_MODEL.pt" --Input "IMAGES_PATH" --OutPut "OUT_PUT_PATH"

        In Windows:  
        !python main.py --Model "YOUR_MODEL.pt" --Input "IMAGES_PATH" --OutPut "OUT_PUT_PATH"
"""

parser = argparse.ArgumentParser(description=Description)
parser.add_argument('--Model', type=str, help='THE MODEL PATH (YOUR_MODEL.pt)')
parser.add_argument('--Input', type=str, help='THE PATH OF INPUT IMAGES')
parser.add_argument('--OutPut', type=str, help='THE PATH FOR SAVE LABELED IMAGES')
args = parser.parse_args()

INPUT_PATH = args.Input
OUTPUT_PATH = args.OutPut
MODEL_PATH = args.Model

IMAGES = os.listdir(INPUT_PATH)
model = YOLO(MODEL_PATH ,verbose=True)
box_an = sv.BoundingBoxAnnotator()
label_an = sv.LabelAnnotator()

def save(image_name:str, img_path:str , detection):
    print("===================================="*2)
    print("Is this detection correct? (y/n) \npress q to stop the program")
    input_ = input(">> ")
    
    if input_.lower() == "y":
        objects = [f"{label} {int(boxs[0])} {int(boxs[1])} {int(boxs[2] - boxs[0])} {int(boxs[3] - boxs[1])}" for label, boxs in zip(detection.class_id, detection.xyxy)]

        with open(os.path.join(OUTPUT_PATH, image_name[:-4] + ".txt"), "w") as f:
            txt_object = "\n".join(objects)

            f.write(txt_object)
            os.rename(img_path, os.path.join(OUTPUT_PATH, image_name))
        print("the image and annotate saved in this path : {}".format(OUTPUT_PATH))

    elif input_.lower() == "n":
        print("Ok")
    
    elif input_.lower() == "q" :
        print("have good day :)")
        exit()

    else:
        print("The input is not valid")
    print("===================================="*2)

for image_name in IMAGES:
    img_path = os.path.join(INPUT_PATH, image_name)
    frame = cv2.imread(img_path)
    
    predict = model.predict(frame)[0]
    detection = sv.Detections.from_ultralytics(predict)
    

    frame = box_an.annotate(frame, detections=detection)
    frame = label_an.annotate(frame , detections=detection ,labels=[f'{label} | {conf:.2}' for label, conf in zip(detection.data["class_name"], detection.confidence)])

    cv2.imshow("frame", frame)
    if cv2.waitKey(100) == ord('q'):
        cv2.destroyAllWindows()
    save(image_name, img_path, detection)
