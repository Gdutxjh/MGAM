# import some common libraries
import numpy as np
import os, json, cv2, random

# Some basic setup:
# Setup detectron2 logger
# import detectron2
# from detectron2.utils.logger import setup_logger
# setup_logger()

# from google.colab.patches import cv2_imshow

# # import some common detectron2 utilities
# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog, DatasetCatalog

# from detectron2.modeling import build_model
# from detectron2.checkpoint import DetectionCheckpointer

# before '4.15.0'
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

# def object_detection():
    
#     cfg = get_cfg()
#     # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
#     cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
#     # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
#     # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
#     cfg.MODEL.WEIGHTS = "../model/detectron/model_final_f10217.pkl"
#     # model = build_model(cfg)
#     # predictor = DetectionCheckpointer(model).load("../model/detectron/model_final_f10217.pkl")

#     predictor = DefaultPredictor(cfg)
    
#     image_path = "../data/root/image"
#     output_dir = "output/data/image_to_text.json"
#     image_names = os.listdir(image_path)
#     m = len(image_names)
#     image_to_text = {}
#     for i in range(m):
        
#         img = cv2.imread(os.path.join(image_path, image_names[i]))
#         outputs = predictor(img)

#         v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
#         out, labels = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#         # print(labels)
#         print("\r image: {}, [{}/{}]".format(image_names[i], i, m), end="")
        
#         image = image_names[i].split(".")[0]
#         image_to_text[image] = []
#         for label in labels:
#             label = label.split(" ")[0]
#             if str(label) not in image_to_text[image]:
#                 image_to_text[image].append(str(label))
#     with open(os.path.join(output_dir), "w") as f:
#         f.write(json.dumps(image_to_text, indent=2))

def image_caption():
    model_path = "../model/vit-gpt2-image-captioning"
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    feature_extractor = ViTImageProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    device = torch.device("cuda")
    model.to(device)
    max_length = 20
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
    
    image_root = "../data/root/image"
    json_path = "output/data/image_to_text.json"
    output_dir = "output/data/image_to_text2.json"
    
    image_names = os.listdir(image_root)
    m = len(image_names)
    image_to_text = json.load(open(json_path))
    
    image_to_text2 = {}
    for key, val in image_to_text.items():
        image_to_text2[key] = {}
        image_to_text2[key]["object"] = val
    
    for i in range(m):
        image_path = os.path.join(image_root, image_names[i])
        
        caption = predict_step([image_path], model, feature_extractor, tokenizer, device, gen_kwargs)
        image = image_names[i].split(".")[0]
        image_to_text2[image]["caption"] = caption
        print("\r [%d/%d]" %(i, m), end="")
        
    with open(os.path.join(output_dir), "w") as f:
        f.write(json.dumps(image_to_text2, indent=2))
    
def predict_step(image_paths, model, feature_extractor, tokenizer, device, gen_kwargs):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds


if __name__ == "__main__":
    image_caption()

