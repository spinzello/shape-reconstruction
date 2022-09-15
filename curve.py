####################################################################################################
    # Imports
####################################################################################################
import numpy as np
import pickle

####################################################################################################
    # Functions
####################################################################################################
def fit(pcd, order=4):
    points = np.asarray(pcd.points)
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    t = np.linspace(0, 1, len(x))

    # Order points according to principal axis
    max_dists = [max(x)-min(x), max(y)-min(y), max(z)-min(z)]
    main_axis = np.argmax(max_dists)
    if main_axis == 0:
        y = y[np.argsort(x)]
        z = z[np.argsort(x)]
        x = np.sort(x)
    elif main_axis == 1:
        x = x[np.argsort(y)]
        z = z[np.argsort(y)]
        y = np.sort(y)
    elif main_axis == 2:
        x = x[np.argsort(z)]
        y = y[np.argsort(z)]
        z = np.sort(z)

    px = np.polyfit(t, x, order)
    py = np.polyfit(t, y, order)
    pz = np.polyfit(t, z, order)
    return [px, py, pz]

def track_points(coeff, point_count=3, distr='equal'):
    array_length = 1000
    t = np.linspace(0,1,array_length)
    X = np.polyval(coeff[0], t)
    Y = np.polyval(coeff[1], t)
    Z = np.polyval(coeff[2], t)
    curve = np.stack((X, Y, Z))

    if distr == 'equal':
        idx = np.round(np.linspace(0, array_length - 1, point_count)).astype(int)
        points = curve[:,idx]
    else:
        idx = np.array(distr) * (array_length-1)
        points = curve[:,idx.astype(int)]

    # Get curve length
    diff = curve - np.roll(curve, shift=1)
    lengths = np.linalg.norm(diff[:,1:], axis=0)
    total_length = np.sum(lengths)
    return points, total_length

def save_points(points):
    with open("output/tracking_points/curve_fit.pkl", "wb") as file:
        pickle.dump(points, file)
    print("")
