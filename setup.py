####################################################################################################
    # Imports
####################################################################################################
import os
import time

import pyrealsense2 as rs
from imports.realsense_device_manager import DeviceManager
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg

####################################################################################################
    # Global variables
####################################################################################################
global DECIMATION
# Select Object (0=person, 39=bottle, 41=cup, 42=fork, 43=knife, 44=spoon, 
# 45=bowl, 58=potted plant, 63:laptop, 64=mouse, 66=keyboard, 67=cell phone, 
# 73=book, 76=scissors, 80=arm)
global OBJECT
DECIMATION = 2
OBJECT = 80

####################################################################################################
    # Functions
####################################################################################################
def cameras(width=848, height=480, fps=60, decimation=DECIMATION):
    print("\nSetting up cameras...")
    # Variables
    resolution_width = width # Available resolutions: 424x240 / 480x270 / 640x360 / 640x480 / 848x480 / 1280x720
    resolution_height = height
    frame_rate = fps  # fps

    config = rs.config()
    config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
    config.enable_stream(rs.stream.infrared, resolution_width, resolution_height, rs.format.y8, frame_rate)
    config.enable_stream(rs.stream.color, int(resolution_width/decimation), int(resolution_height/decimation), rs.format.bgr8, frame_rate)

    device_manager = DeviceManager(rs.context(), config)
    device_manager.enable_all_devices()
    device_manager.load_settings_json("calibration/mobilerack_preset.json")
    device_manager.enable_emitter(on=True, always=False, max_power=True)
    

    if len(device_manager._available_devices) == 0:
        print("Please connect a device.")
        exit()

    # Allow some frames for the auto-exposure controller to stablise
    dispose_frames_for_stablisation = 30  # frames
    for frame in range(dispose_frames_for_stablisation):
        frames = device_manager.poll_frames()
        
    print("Cameras are set up!\n")
    return device_manager

def detectron():
    # Register the desired dataset
    dataset_name="sopra_train"
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

    MODEL_WEIGHTS = os.path.join(ROOT_DIR, "{0}/trained_model/model_final.pth".format(dataset_name))
    cfg.MODEL.WEIGHTS = MODEL_WEIGHTS

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8   # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 81  # 
    cfg.TEST.DETECTIONS_PER_IMAGE = 100
    return cfg