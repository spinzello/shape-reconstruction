####################################################################################################
    # Imports
####################################################################################################
import os
import random

import cv2
import numpy as np

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

####################################################################################################
    # Main Function
####################################################################################################
def main():
    print("Start")
    predictor = DefaultPredictor(set_cfg())
    img_directory = "/home/seb/Datasets/sopra_test/images"
    save_img_directory = "/home/seb/Datasets/sopra_test/segmented_images"
    img_list = os.listdir(img_directory)
    random.shuffle(img_list)
    count = 0
    for img_name in img_list:
        print(count)
        img = cv2.imread(os.path.join(img_directory, img_name))
        segmented_img = segment_image(img, predictor)
        cv2.imwrite(os.path.join(save_img_directory, img_name), segmented_img)
        count += 1
    print("Done")

####################################################################################################
    # Auxiliary Functions
####################################################################################################
def set_cfg():
    # Register the desired dataset
    dataset_name="sopra_test"
    ROOT_DIR = '/home/seb/Datasets/'
    IMAGE_DIR = os.path.join(ROOT_DIR, "{}/images".format(dataset_name))
    JSON = os.path.join(ROOT_DIR, "{0}/{0}.json".format(dataset_name))
    register_coco_instances(dataset_name, {}, JSON, IMAGE_DIR)

    # Add Metadata to dataset
    id_mapping = MetadataCatalog.get("coco_2017_val").thing_dataset_id_to_contiguous_id.copy()
    id_mapping[91] = 80
    thing_classes = MetadataCatalog.get("coco_2017_val").thing_classes.copy()
    thing_classes.append('arm')
    MetadataCatalog.get(dataset_name).set(thing_classes = thing_classes)
    MetadataCatalog.get(dataset_name).set(thing_dataset_id_to_contiguous_id = id_mapping)

    # Construct config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TEST = (dataset_name, )

    cfg.MODEL.DEVICE = "cuda"
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.WEIGHTS =  '/home/seb/Documents/master_thesis/main/sopra_segmentation_model/model_0002759.pth'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 81  # 
    cfg.TEST.DETECTIONS_PER_IMAGE = 100
    return cfg

def segment_image(img, predictor):
    my_metadata = MetadataCatalog.get("sopra_test")

    # Segment image   
    prediction = predictor(img)
    instances = prediction["instances"].to('cpu')

    # Visualize segmentation masks
    vis_color = Visualizer(img, metadata=my_metadata) #
    vis_color_output = vis_color.draw_instance_predictions(predictions=instances)
    img = cv2.cvtColor(vis_color_output.get_image()[:, :, ::-1], cv2.COLOR_RGB2BGR)
    return img


if __name__ == "__main__":
    main()