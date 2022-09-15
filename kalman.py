####################################################################################################
    # Imports
####################################################################################################
import numpy as np
import cv2

####################################################################################################
    # Functions
####################################################################################################
def initialize(devices, frames):
    frame_size = frames[0][:,:,0].size
    initial_state = []
    initial_covariance = []

    for ID in devices._available_devices:
        initial_state.append([np.zeros((frame_size)), np.zeros((frame_size))])
        initial_covariance.append([np.ones((frame_size)), np.zeros((frame_size)), np.ones((frame_size))])
    
    return initial_state, initial_covariance

def apply(masks, states, covariances, confidences, verbose=False):
    new_states = []
    new_covariances = []
    filtered_masks = []
    frame_size = masks[0].shape
    mask_threshold = 100
    r_100 = 0.0 # Noise variance at 100% confidence
    r_90 = 1
    r_50 = 10
    Q = 1.0
    
    for mask, x, P, confidence in zip(masks, states, covariances, confidences):
        # Calculate measurement noise
        conf = 1 - confidence
        a = 10*r_50 - 250*r_90 + 240*r_100
        b = 125*r_90 - r_50 - 124*r_100
        R =  a*conf**3 + b*conf**2 + r_100

        # Prediction step
        x_p = [2*x[0] - x[1], x[0]]
        P_p = [4*(P[0] - P[1]) + P[2] + Q, 2*P[0] - P[1], P[0] + Q] # Predicted covariance matrix is symmetric, therefore size(P_p) = 3

        # Measurement update
        z = np.ndarray.flatten(np.float64(mask))
        K = [P_p[0]/(P_p[0] + R), P_p[1]/(P_p[0] + R)]
        x_m     = [x_p[0] + K[0]*(z - x_p[0]), x_p[1] + K[1]*(z - x_p[0])]
        P_m = [(np.ones((z.size)) - K[0])*P_p[0], (np.ones((z.size)) - K[0])*P_p[1], P_p[2] - K[1]*P_p[1]] # Updated covariance matrix is symmetric, therefore size(P_m) = 3

        # Threshold filtered frame to remove negative and values over 255
        filtered_mask = np.int16( np.reshape(x_m[0], (frame_size)) )
        filtered_mask[filtered_mask > mask_threshold] = 255
        filtered_mask[filtered_mask < mask_threshold+1] = 0
        filtered_mask = np.uint8(filtered_mask)

        # Process filtered mask
        filtered_mask = cv2.dilate(filtered_mask, kernel=np.ones((3,3),np.uint8), iterations=3)
        filtered_mask = cv2.erode(filtered_mask, kernel=np.ones((3,3),np.uint8), iterations=3)

        # Append to output list
        new_states.append(x_m)
        new_covariances.append(P_m)
        filtered_masks.append(filtered_mask)
        
        if verbose == True:
            print('x_m[0]:', "%.6f" % np.min(x_m[0]), '/',"%.6f" % np.max(x_m[0]), '/',"%.6f" % np.sum(x_m[0]), '||',
                'P_m[0]:', "%.6f" % np.min(P_m[0]), '/',"%.6f" % np.max(P_m[0]), '||',
                'P_m[1]:', "%.6f" % np.min(P_m[1]), '/',"%.6f" % np.max(P_m[1]), '||',
                'K[0]:', "%.6f" % np.min(K[0]), '/',"%.6f" % np.max(K[0]), '||',
                'R:', "%.6f" % R, '||',
                'Confidence:', "%.6f" % confidence, '||')

    return filtered_masks, new_states, new_covariances