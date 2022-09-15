import os
import torch
import json

from detectron2.data.build import build_detection_train_loader
from detectron2.data.catalog import DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets import load_coco_json
from detectron2.data import MetadataCatalog
from detectron2.data import transforms as T
from detectron2.data import DatasetMapper

def main():
    # Get metadata from COCO & combine with arm dataset
    id_mapping = MetadataCatalog.get("coco_2017_val").thing_dataset_id_to_contiguous_id.copy()
    id_mapping[91] = 80
    thing_classes = MetadataCatalog.get("coco_2017_val").thing_classes.copy()
    thing_classes.append('arm')

    # Register new dataset with combined metadata
    register_dataset('sopra_test', thing_classes, id_mapping)
    register_dataset('sopra_train', thing_classes, id_mapping)

    train()


def register_dataset(dataset_name, thing_classes, id_mapping):
    ROOT_DIR = '/home/seb/Datasets/'
    IMAGE_DIR = os.path.join(ROOT_DIR, "{}/images".format(dataset_name))
    JSON = os.path.join(ROOT_DIR, "{0}/{0}.json".format(dataset_name))

    # Register & add new metadata
    register_coco_instances(dataset_name, {}, JSON, IMAGE_DIR)
    # MetadataCatalog.get(dataset_name).set(json_file=JSON, image_root=IMAGE_DIR, thing_classes=thing_classes, thing_dataset_id_to_contiguous_id = id_mapping)


def train():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.DEVICE = "cpu"
    cfg.DATASETS.TRAIN = ("sopra_train",)
    cfg.DATASETS.TEST = ("sopra_test",)
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 5000
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.STEPS = [1000,]
    cfg.SOLVER.CHECKPOINT_PERIOD = 10

    # cfg.TEST.EVAL_PERIOD = 10

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 81  # number of thing classes
    cfg.OUTPUT_DIR = '/home/seb/Documents/master_thesis/main/sopra_segmentation_model'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Create dataloader to enable augmentations
    dataloader = custom_dataloader(cfg)

    trainer = DefaultTrainer(cfg)
    trainer.build_train_loader = dataloader
    trainer.resume_or_load(resume=False)
    trainer.build_writers()
    trainer.train()


def custom_dataloader(cfg):
    mapper=DatasetMapper(cfg, is_train=True, augmentations=[
        T.RandomBrightness(0.9, 1.1),
        T.RandomRotation(angle=[0, 360])
        ])
    dataloader = build_detection_train_loader(cfg, mapper=mapper)
    return dataloader


def check_torch_version():
    print("Pytorch version：")
    print(torch.__version__)
    print("CUDA Version: ")
    print(torch.version.cuda)
    print("cuDNN version is :")
    print(torch.backends.cudnn.version())
    print("Arch version is :")
    print(torch._C._cuda_getArchFlags())


if __name__ == "__main__":
    main()
    print('Finished')