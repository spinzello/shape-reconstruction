####################################################################################################
    # Imports
####################################################################################################
import os
import cv2
import torch
import random
from detectron2.data.catalog import DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import load_coco_json
from detectron2.data.datasets import register_coco_instances
from detectron2.data import build_detection_test_loader
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.evaluation import DatasetEvaluators
from detectron2.evaluation import COCOEvaluator
from detectron2.evaluation import inference_on_dataset
from detectron2.utils.visualizer import ColorMode, Visualizer

####################################################################################################
    # Main Function
####################################################################################################
# export DETECTRON2_DATASETS='/home/seb/Datasets'

ROOT_DIR = "/home/seb/Documents/master_thesis/main"

def main():
    dataset = "sopra_test" # "custom_coco_2017_val"
    weights_retrained = '/home/seb/Documents/master_thesis/main/sopra_segmentation_model/model_final.pth'
    weights_pretrained = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    # Get metadata from COCO & combine with arm dataset
    id_mapping = MetadataCatalog.get("coco_2017_val").thing_dataset_id_to_contiguous_id.copy()
    id_mapping[91] = 80
    thing_classes = MetadataCatalog.get("coco_2017_val").thing_classes.copy()
    thing_classes.append('arm')

    # Register new dataset with combined metadata
    register_dataset(dataset, thing_classes, id_mapping)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TEST = (dataset, )

    cfg.MODEL.DEVICE = "cuda"
    cfg.DATALOADER.NUM_WORKERS = 1

    MODEL_WEIGHTS = os.path.join(ROOT_DIR, "sopra_segmentation_model/model_final.pth")
    cfg.MODEL.WEIGHTS = MODEL_WEIGHTS

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4   # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 81  # 
    cfg.TEST.DETECTIONS_PER_IMAGE = 100
    predictor = DefaultPredictor(cfg)

    dataloader = build_detection_test_loader(cfg, dataset)
    evaluator = COCOEvaluator(dataset_name=dataset, output_dir='/home/seb/Documents/master_thesis/main/detectron_output')
    eval_results = inference_on_dataset(predictor.model, dataloader, evaluator)

####################################################################################################
    # Auxiliary Functions
####################################################################################################
def evaluate_standard_coco():
    dataset = "coco_2017_val"
    # export DETECTRON2_DATASETS='/home/seb/Datasets'

    cfg = get_cfg()
    cfg.DATASETS.TEST = (dataset,)
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)

    dataloader = build_detection_test_loader(cfg, dataset)
    evaluator = COCOEvaluator(dataset_name=dataset, output_dir='/home/seb/Documents/master_thesis/main/detectron_output')
    eval_results = inference_on_dataset(predictor.model, dataloader, evaluator)

def register_dataset(dataset_name, thing_classes, id_mapping):
    ROOT_DIR = '/home/seb/Datasets/'
    IMAGE_DIR = os.path.join(ROOT_DIR, "{}/images".format(dataset_name))
    JSON = os.path.join(ROOT_DIR, "{0}/{0}.json".format(dataset_name))

    # Register & add new metadata
    register_coco_instances(dataset_name, {}, JSON, IMAGE_DIR)
    # MetadataCatalog.get(dataset_name).set(thing_classes = thing_classes)
    # MetadataCatalog.get(dataset_name).set(thing_dataset_id_to_contiguous_id = id_mapping)

if __name__ == "__main__":
    main()
    print('Finished')