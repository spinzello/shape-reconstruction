####################################################################################################
    # Imports
####################################################################################################
import os
import json
import numpy as np
from skimage import measure
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon
from collections import defaultdict

####################################################################################################
    # Main Function
####################################################################################################
SET_LOCATION = "/home/seb/Datasets/sopra_train"
IMAGE_LOCATION = "images"
MASK_LOCATION = "masks"

set_name = "sopra_train"
descriptive_name = "SoPrA Train Set"

def main():
    images = sorted(os.listdir(os.path.join(SET_LOCATION, IMAGE_LOCATION)))
    masks = sorted(os.listdir(os.path.join(SET_LOCATION, MASK_LOCATION)))
    annotations = defaultdict(list)
    count = 0
    for image_name, mask_name in zip(images, masks):
        print(count, "/", len(images))
        image = np.asarray(Image.open(os.path.join(SET_LOCATION, IMAGE_LOCATION, image_name)))
        mask = np.asarray(Image.open(os.path.join(SET_LOCATION, MASK_LOCATION, mask_name)))
        image_id = count
        category_id = 91
        annotation_id = count
        is_crowd = 0
        annotation = create_sub_mask_annotation(image, mask, image_id, category_id, annotation_id, image_name)
        # annotations.append(annotation)
        for key, value in annotation.items():
            if count == 0:
                if key == 'info' or key == 'categories':
                    annotations[key] = value
            else:
                if key == 'images' or key == 'annotations':
                    annotations[key].append(value)
            print()
        count += 1

    with open(os.path.join(SET_LOCATION, set_name + '.json'), 'w') as f:
        json.dump(annotations, f)
    return

####################################################################################################
    # Auxiliary Functions
####################################################################################################
def create_sub_mask_annotation(image, mask, image_id, category_id, annotation_id, file_name):

    contours = measure.find_contours(mask, 0.5, positive_orientation='low')

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.extend(segmentation)

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    area = multi_poly.area

    annotation = {
        'info':
            {
                'description': descriptive_name
            },
        'images':
            {
                'id': image_id,
                'file_name': file_name,
                'height': image.shape[0],
                'width': image.shape[1]
            },
        'categories':
            [
                
                {"supercategory": "person","id": 1,"name": "person"},
                {"supercategory": "vehicle","id": 2,"name": "bicycle"},
                {"supercategory": "vehicle","id": 3,"name": "car"},
                {"supercategory": "vehicle","id": 4,"name": "motorcycle"},
                {"supercategory": "vehicle","id": 5,"name": "airplane"},
                {"supercategory": "vehicle","id": 6,"name": "bus"},
                {"supercategory": "vehicle","id": 7,"name": "train"},
                {"supercategory": "vehicle","id": 8,"name": "truck"},
                {"supercategory": "vehicle","id": 9,"name": "boat"},
                {"supercategory": "outdoor","id": 10,"name": "traffic light"},
                {"supercategory": "outdoor","id": 11,"name": "fire hydrant"},
                {"supercategory": "outdoor","id": 13,"name": "stop sign"},
                {"supercategory": "outdoor","id": 14,"name": "parking meter"},
                {"supercategory": "outdoor","id": 15,"name": "bench"},
                {"supercategory": "animal","id": 16,"name": "bird"},
                {"supercategory": "animal","id": 17,"name": "cat"},
                {"supercategory": "animal","id": 18,"name": "dog"},
                {"supercategory": "animal","id": 19,"name": "horse"},
                {"supercategory": "animal","id": 20,"name": "sheep"},
                {"supercategory": "animal","id": 21,"name": "cow"},
                {"supercategory": "animal","id": 22,"name": "elephant"},
                {"supercategory": "animal","id": 23,"name": "bear"},
                {"supercategory": "animal","id": 24,"name": "zebra"},
                {"supercategory": "animal","id": 25,"name": "giraffe"},
                {"supercategory": "accessory","id": 27,"name": "backpack"},
                {"supercategory": "accessory","id": 28,"name": "umbrella"},
                {"supercategory": "accessory","id": 31,"name": "handbag"},
                {"supercategory": "accessory","id": 32,"name": "tie"},
                {"supercategory": "accessory","id": 33,"name": "suitcase"},
                {"supercategory": "sports","id": 34,"name": "frisbee"},
                {"supercategory": "sports","id": 35,"name": "skis"},
                {"supercategory": "sports","id": 36,"name": "snowboard"},
                {"supercategory": "sports","id": 37,"name": "sports ball"},
                {"supercategory": "sports","id": 38,"name": "kite"},
                {"supercategory": "sports","id": 39,"name": "baseball bat"},
                {"supercategory": "sports","id": 40,"name": "baseball glove"},
                {"supercategory": "sports","id": 41,"name": "skateboard"},
                {"supercategory": "sports","id": 42,"name": "surfboard"},
                {"supercategory": "sports","id": 43,"name": "tennis racket"},
                {"supercategory": "kitchen","id": 44,"name": "bottle"},
                {"supercategory": "kitchen","id": 46,"name": "wine glass"},
                {"supercategory": "kitchen","id": 47,"name": "cup"},
                {"supercategory": "kitchen","id": 48,"name": "fork"},
                {"supercategory": "kitchen","id": 49,"name": "knife"},
                {"supercategory": "kitchen","id": 50,"name": "spoon"},
                {"supercategory": "kitchen","id": 51,"name": "bowl"},
                {"supercategory": "food","id": 52,"name": "banana"},
                {"supercategory": "food","id": 53,"name": "apple"},
                {"supercategory": "food","id": 54,"name": "sandwich"},
                {"supercategory": "food","id": 55,"name": "orange"},
                {"supercategory": "food","id": 56,"name": "broccoli"},
                {"supercategory": "food","id": 57,"name": "carrot"},
                {"supercategory": "food","id": 58,"name": "hot dog"},
                {"supercategory": "food","id": 59,"name": "pizza"},
                {"supercategory": "food","id": 60,"name": "donut"},
                {"supercategory": "food","id": 61,"name": "cake"},
                {"supercategory": "furniture","id": 62,"name": "chair"},
                {"supercategory": "furniture","id": 63,"name": "couch"},
                {"supercategory": "furniture","id": 64,"name": "potted plant"},
                {"supercategory": "furniture","id": 65,"name": "bed"},
                {"supercategory": "furniture","id": 67,"name": "dining table"},
                {"supercategory": "furniture","id": 70,"name": "toilet"},
                {"supercategory": "electronic","id": 72,"name": "tv"},
                {"supercategory": "electronic","id": 73,"name": "laptop"},
                {"supercategory": "electronic","id": 74,"name": "mouse"},
                {"supercategory": "electronic","id": 75,"name": "remote"},
                {"supercategory": "electronic","id": 76,"name": "keyboard"},
                {"supercategory": "electronic","id": 77,"name": "cell phone"},
                {"supercategory": "appliance","id": 78,"name": "microwave"},
                {"supercategory": "appliance","id": 79,"name": "oven"},
                {"supercategory": "appliance","id": 80,"name": "toaster"},
                {"supercategory": "appliance","id": 81,"name": "sink"},
                {"supercategory": "appliance","id": 82,"name": "refrigerator"},
                {"supercategory": "indoor","id": 84,"name": "book"},
                {"supercategory": "indoor","id": 85,"name": "clock"},
                {"supercategory": "indoor","id": 86,"name": "vase"},
                {"supercategory": "indoor","id": 87,"name": "scissors"},
                {"supercategory": "indoor","id": 88,"name": "teddy bear"},
                {"supercategory": "indoor","id": 89,"name": "hair drier"},
                {"supercategory": "indoor","id": 90,"name": "toothbrush"},
                {"supercategory": "indoor","id": 91,"name": "arm"}
                
            ],
        'annotations': 
            {
                'segmentation': [segmentations],
                'iscrowd': 0,
                'image_id': image_id,
                'category_id': category_id,
                'id': annotation_id,
                'bbox': bbox,
                'area': area
            }        
    }

    return annotation

if __name__ == "__main__":
    main()