####################################################################################################
    # Imports
####################################################################################################
import numpy as np
import itertools

import setup
from imports.helper_functions import convert_depth_pixel_to_metric_coordinate
####################################################################################################
    # Functions
####################################################################################################
def run(color_frames,
              depth_frames,
              predictor,
              prev_object_position,
              prev_object_count,
              calibration_info_devices):

    object_count = []
    all_box_centers = []
    all_scores = []
    all_masks = []

    # Extract segmentation info (masks, box centers, confidences, object count)
    for color_frame in color_frames:
        instances = predictor(color_frame)["instances"].to('cpu')
        is_object = instances._fields['pred_classes'] == setup.OBJECT
        object_count.append(instances._fields['pred_classes'][is_object].size()[0])
        all_box_centers.append(np.array(instances._fields['pred_boxes'][is_object].get_centers()))
        all_scores.append(instances._fields['scores'][is_object])
        all_masks.append(instances._fields['pred_masks'][is_object])

    masks = []
    confidences = []
    object_position_samples = []

    if sum(object_count) == 0:
        print("- No object found -\n")
        for _ in color_frames:
            masks.append(np.zeros((color_frames[0].shape[:2]), dtype='u1'))
            confidences.append(0)
            object_position = None
        return masks, confidences, object_position, object_count
    else:
        if (prev_object_position is None): # or (object_count != prev_object_count):
            # Find optimal mask configuration that minimizes 3D distance between objects
            print("- Found new object -\n")

            # Find all object coordinates
            all_object_positions = []
            object_indices = []
            for box_centers, depth_frame, mask_candidates, calibration_info, objects_recognized in zip(all_box_centers, depth_frames, all_masks, calibration_info_devices, object_count):
                if objects_recognized != 0:
                    all_object_positions.append(pixel_to_world_point(box_centers.astype('u2'), depth_frame, mask_candidates, calibration_info)) # TODO: use average depth of all the masks pixels instead of depth at the center position (center doesn't need to be on the object)
                    object_indices.append(list(np.arange(box_centers.shape[0])))

            if len(all_object_positions) > 1:
                # Evaluate distance between all possible combinations of objects
                all_combinations = np.array(list(itertools.product(*all_object_positions)))
                all_combinations_indices = list(itertools.product(*object_indices))
                all_combinations = np.append(all_combinations, all_combinations[:,0,None], axis=1) # Extend array by the first row so that when the difference is calculated, the difference between the first and last row is taken into account
                distances = np.sum( np.linalg.norm( np.diff(all_combinations, axis=1), ord=2, axis=2), axis=1) # TODO: disqualify combinations that surpass a distance threshold

                # Find best match
                lowest_distance_index = np.argmin(distances)
                best_combination = list(all_combinations_indices[lowest_distance_index])
            else:
                # Choose mask with highest confidence
                scores = [scores for scores, count in zip(all_scores, object_count) if count != 0]
                best_combination = [np.argmax(scores[0]).item()]

            # Return parameters of selected object
            for mask_candidates, scores, count in zip(all_masks, all_scores, object_count):
                if count == 0:
                    masks.append(np.zeros((color_frames[0].shape[:2]), dtype='u1'))
                    confidences.append(0)
                else:
                    masks.append(np.asanyarray(mask_candidates[best_combination[0]]).astype('u1'))
                    object_position_samples.append(all_object_positions[0][best_combination[0]])
                    confidences.append(scores[best_combination[0]].item())

                    best_combination.pop(0)
                    all_object_positions.pop(0)
        else:
            for mask_candidates, box_centers, scores, depth_frame, calibration_info in zip(all_masks, all_box_centers, all_scores, depth_frames, calibration_info_devices):
                if scores.numel() == 0:
                    masks.append(np.zeros((color_frames[0].shape[:2]), dtype='u1'))
                    confidences.append(0)
                else:
                    # Append mask that has its center closest to the other masks in 3D)
                    object_position_candidates = pixel_to_world_point(box_centers.astype('u2'), depth_frame, mask_candidates, calibration_info)
                    distances_to_prev_object = np.linalg.norm((object_position_candidates - prev_object_position), ord=2, axis=1)
                    closest_object_index = int(np.argmin(distances_to_prev_object))
                    object_position_samples.append(object_position_candidates[closest_object_index])
                    masks.append(np.asanyarray(mask_candidates[closest_object_index]).astype('u1'))    
                    confidences.append(scores[closest_object_index].item())

        object_position = np.mean(np.array(object_position_samples), axis=0)
        masks = [mask*255 for mask in masks]

        return masks, confidences, object_position, object_count

def pixel_to_world_point(pixel_coordinates_array, depth_frame, mask_candidates, calibration):
    points_3D = []
    for pixel_coordinates, mask in zip(pixel_coordinates_array, mask_candidates):
        masked_depth_frame = depth_frame[mask]
        depth = np.mean(masked_depth_frame[masked_depth_frame != 0])
        # depth = depth_frame[pixel_coordinates[1], pixel_coordinates[0]]
        point_3D = convert_depth_pixel_to_metric_coordinate(depth, 
                                                            pixel_coordinates[0],
                                                            pixel_coordinates[1],
                                                            calibration[1]) # pixel x, pixel y
        point_3D = np.array( [ [point_3D[0]/10000], [point_3D[1]/10000], [point_3D[2]/10000] ] )
        # Transform point-cloud's origin from RGB sensor to depth sensor (Requires extrinsics)
        # point_3D = calibration[2].apply_transformation(point_3D)
        # Transform point-cloud to world coordinates (Requires world frame transformation)
        point_3D = calibration[0].apply_transformation(point_3D)
        points_3D.append(np.transpose(point_3D)[0])
    return np.array(points_3D)