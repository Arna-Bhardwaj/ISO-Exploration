import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import math
import matplotlib.colors as mcolors

# t1 = np.array([[134, -203, -108, 3.6429358973488144], [227, 191, 174, 1.5417254142425594], [-134, 159, 137, 1.510210065119609], [-160, 216, -165, 1.0841289766564353], [148, -214, -244, 4.743515947635559], [-221, 115, -218, 2.4971628191080644]])
# t1 = np.array([[-100, -100, 100]])

# t2 = [([[ 100.23380344, -110.86412762,  292.07454347,    3.37366674]]), ([[152.74993849, -84.3751336 ,  64.63475588,   2.66197468],
#         [49.41467128, 623.50521311, 141.57094836,   3.81983348]]), ([[ 223.21313167, -232.79158249, -135.37186543,    3.13385562],
#         [117.11851419, -187.88648121, -134.27830741,    4.7132869] ,
#        [-183.62674696, -126.79343401,  138.64620912,    1.09180301]]), ([[ 5.07261945e+02,  5.59752085e+01, -3.15496310e+02,  3.43011499e+00],
#         [7.43663390e+01,  4.45597842e+01,  1.24299628e+02,  4.25426235e+00],
#        [-1.24995186e+02,  2.67503232e+02,  3.01483683e+02,  5.53317182e+00],
#        [-2.81914367e+02,  3.21252535e+02, -8.32732181e+02,  1.73883175e-01]]), ([[-179.22640025, -287.53227547, -214.49827124,    4.18560672],
#        [-118.12486182, -167.70484023,  -97.98522743,    1.65342148],
#        [-198.47138089,  246.8284196 ,  348.62765389,    2.49401423],
#         [224.561441  , -310.04996568,  129.21860235,    0.85968819],
#         [108.74247769,  -98.6178536 ,  291.05174038,    6.04754237]]), ([[-4.41892461e+02, -7.52925191e+01,  5.99586732e+02,  5.75562591e+00],
#        [-1.21792939e+03, -1.29562274e+02, -1.95075365e+02,  8.67992190e-01],
#         [4.72495610e+02, -5.37009400e+01, -8.79076763e+01,  0.00000000e+00],
#         [2.84647223e+01,  5.59126826e+02,  4.75703702e+02,  1.69165441e+00],
#         [6.91762969e+02, -1.21568119e+02,  2.94347035e+02,  2.51568799e+00],
#        [-7.08065861e+02, -4.40905995e+01, -1.07713642e+03,  3.38570168e+00]]), ([[ 4.66487683e+02,  7.02736044e+01, -2.50149219e+02,  4.46703410e+00],
#         [4.64688376e+02,  5.73235455e+02,  1.00321754e+03,  5.29851180e+00],
#         [2.75855469e+02,  1.61224858e+02,  1.21171855e+02,  2.47178406e+00],
#         [1.82348103e+02, -1.62253180e+02, -1.59604904e+01,  4.69208403e-03],
#        [-8.42325242e+01,  3.29688086e+02,  1.18823673e+02,  7.93382389e-01],
#        [-8.96425389e+01, -3.27420320e+02, -1.79789479e+02,  3.26947030e+00],
#        [-1.02946524e+01,  4.06710164e+00,  1.11647380e+01,  6.08463334e+00]])]

# t3 = [([[  70.50875822, -132.08989547,  269.95139452,    5.43412295]]), ([[-115.24905595, -239.02576527,  -66.10035096,    5.09760526],
#         [110.94567324, -104.54621026,  301.63374845,    6.08535573]]), ([[ 148.82487155, -187.16809268, -240.82046692,    4.39267817],
#        [-131.48109735,  220.89821162,  214.74728047,    5.25110476],
#         [ 19.17254882,  163.60314778, -164.43332697,    2.89105617]]), ([[ 114.97512816,  105.47709925, -224.36718072,    2.29248519],
#        [-194.82556753,  167.59180141,  260.19372849,    4.82451689],
#         [147.49213022, -159.64061774, -146.60874557,    5.89357075],
#        [-165.42286287, -116.15738244, -125.46152999,    3.37953549]]), ([[-122.17907611,  178.01692639,  131.16419602,    1.12739908],
#         [164.9319732 , -184.24679274,  225.39248177,    4.33202518],
#        [-244.75377095,  200.32071127, -171.72728396,    6.16341498],
#         [140.24372606,  203.70222728,  159.03098277,    5.2432854],
#         [194.52052486,  155.5413729 ,  124.938265  ,    3.24803307]]), ([[ 132.52826571, -114.24655569,  112.37006732,    3.62751102],
#        [-221.14630311,  135.40737177,  230.36779245,    4.51353071],
#        [-267.67638053, -216.32913228,  -95.00289393,    5.33346316],
#         [216.64856876, -237.83691078, -158.61794758,    1.70900049],
#        [-134.59406015,  180.46605345,  105.51308382,    0.88539276],
#         [218.40096038,  170.82538929,  144.81943085,    6.13503295]]), ([[ 2.28438194e+02, -9.18408817e+01, -2.69802828e+02,  3.66113595e+00],
#        [-1.58841920e+02,  1.21928471e+02, -8.50460071e+01,  2.80247436e+00],
#         [1.71108161e+02, -5.87385291e+01, -2.07044036e+02,  1.97131259e+00],
#        [-2.85440341e+02, -2.49464722e+02,  6.57506746e+02,  6.28211293e+00],
#        [-1.69334926e+02, -2.12148199e+02,  1.68161677e+02,  2.31732200e-02],
#        [-8.10571100e+01,  3.17896915e+02,  2.86668294e+02,  5.45268012e+00],
#        [-7.56255016e+02, -5.45431192e+02, -5.56399777e+02,  4.58672838e+00]])]


# t4 = [([[-311.18484314, -239.98553706, -467.37949359,    1.95984805]]),
#  ([[ 4.67005901e+02, -2.86026790e+02,  7.75562604e+01,  5.79679798e-03],
#          [3.34430603e+01, -2.84090173e+02,  1.59615802e+02,  9.17495594e-01]]),
#  ([ [512.14077576, -283.25901924, -483.90205944,    5.44196187],
#          [349.69908217, -171.43083135, -548.52790012,    2.17366156],
#         [-494.35041249, -268.32419312, -782.34210202,    6.27941442]]),
#  ([ [219.27577115, -159.79733232,  469.70876978,    4.82443617],
#          [253.65942005, -142.57460908,  499.68966252,    4.0390375],
#          [236.95692324, -158.172057  ,  528.25236132,    6.28318491],
#          [247.3350109 , -162.06572995,  480.22018376,    5.60983367]]),
#  ([[-5.59188661e+02,  3.20864486e+02,  6.97547285e+02,  1.78355269e+00],
#         [-9.66252370e+01,  3.50190863e+02,  7.26028633e+02,  4.34725838e+00],
#         [-1.40488250e+02,  2.98905884e+02, -1.09523832e+02,  9.25357637e-01],
#         [-1.34195248e+03,  8.64266810e+01, -5.17452015e+02,  3.23349806e+00],
#         [-1.08924417e+03, -3.38737956e+02,  5.60796149e+02,  9.32713955e-02]]),
#  ([ [339.        , -405.        ,  156.        ,    5.53561156],
#          [339.        , -405.        ,  156.        ,    5.53561156],
#          [339.        , -405.        ,  156.        ,    5.81239213],
#          [339.        , -405.        ,  156.        ,    5.53561156],
#          [339.        , -405.        ,  156.        ,    5.53561156],
#          [339.        , -405.        ,  156.        ,    5.53561156]]),
#  ([ [470.50714286, -391.08392857, -541.8875    ,    3.81284231],
#          [470.50714286, -391.08392857, -541.8875    ,    3.81284231],
#          [470.50714286, -391.08392857, -541.8875    ,    3.81284231],
#          [470.50714286, -391.08392857, -541.8875    ,    3.81284231],
#          [470.50714286, -391.08392857, -541.8875    ,    3.81284231],
#          [470.50714286, -391.08392857, -541.8875    ,    3.81284231],
#          [421.2       , -391.08392857, -541.8875    ,    3.81284231]]),
# ]
# t2 = [([[-122.17907611,  178.01692639,  131.16419602,    1.12739908],
#         [164.9319732 , -184.24679274,  225.39248177,    4.33202518],
#        [-244.75377095,  200.32071127, -171.72728396,    6.16341498],
#         [140.24372606,  203.70222728,  159.03098277,    5.2432854],
#         [194.52052486,  155.5413729 ,  124.938265  ,    3.24803307]])]

t2 = [([[-122.17907611,  178.01692639,  131.16419602,    1.12739908],
        [164.9319732 , -184.24679274,  225.39248177,    4.33202518],
       [-244.75377095,  200.32071127, -171.72728396,    6.16341498],
        [140.24372606,  203.70222728,  159.03098277,    5.2432854],
        [194.52052486,  155.5413729 ,  124.938265  ,    3.24803307]])]
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 18


fov = np.pi / 4

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

r = 0.05
u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
x_rad, y_rad, z_rad = 100, 100, 100
x_sphere = x_rad * np.cos(u) * np.sin(v)
y_sphere = y_rad * np.sin(u) * np.sin(v)
z_sphere = z_rad * np.cos(v)
ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.2, color='white', label = "Uncertainty Ellipsoid")

x_rad, y_rad, z_rad = 100, 100, 100
a, b, c = x_rad, y_rad, z_rad

# Function to generate a random point inside a unit sphere using rejection sampling
def randompoint(ndim=3):
    while True:
        point = np.random.uniform(-1, 1, size=(ndim,))
        if np.linalg.norm(point) <= 1:
            return point

# Function to scale points from the unit sphere to the ellipsoid
f = lambda x, y, z: np.multiply(np.array([a, b, c]), np.array([x, y, z]))

# Generate points until we have, let's say, 5000 points
n = 5000
points = []
while len(points) < n:
    x, y, z = randompoint()
    scaled_point = f(x, y, z)
    points.append(scaled_point)

# Convert list to numpy array for easier handling
points = np.array(points)
# ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)

def is_point_in_cone(point, apex, axis, height=math.sqrt(5.75*2 + 5.75**2 + 5.75**2), radius=3):
    apex_to_point = point - apex
    projection_length = np.dot(apex_to_point, axis)
    if projection_length < 0 or projection_length > height:
        return False
    cone_radius_at_height = (projection_length / height) * radius
    perpendicular_distance = np.linalg.norm(apex_to_point - projection_length * axis)
    return perpendicular_distance <= cone_radius_at_height

def rotate_vector(vector, axis, angle):
    axis = axis / np.linalg.norm(axis)
    vector = vector * np.cos(angle) + np.cross(axis, vector) * np.sin(angle) + axis * np.dot(axis, vector) * (1 - np.cos(angle))
    return vector


# Define the position of the apex of the cone (green X)
for t in t2:
    t1 = np.array(t)
    obs_POIs = []
    non_obs = []
    print(np.shape(t1))
    for sc in t1:
        height = math.sqrt(sc[0]**2 + sc[1]**2 + sc[2]**2)
        radius = height * math.tan(fov / 2)
        resolution = 100

        theta = np.linspace(0, 2*np.pi, resolution)
        z = np.linspace(0, height, resolution)
        theta, z = np.meshgrid(theta, z)
        r = radius * (height - z) / height
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        apex = np.array([0, 0, 0])
        origin = np.array([sc[0], sc[1], sc[2]])
        distance = np.linalg.norm(origin - apex)

        # Translate the cone to have its apex at the specified position
        x += apex[0]
        y += apex[1]
        z += apex[2]

        # Calculate the direction vector from the apex to the origin
        direction_vector = origin - apex
        vec = origin - apex
        direction_vector = direction_vector / np.linalg.norm(direction_vector)

        # Calculate the axis and angle of rotation to align the cone's axis with the direction vector
        axis_of_rotation = np.cross([0, 0, 1], direction_vector)
        angle_of_rotation = np.arccos(np.dot([0, 0, 1], direction_vector))

        # Apply rotation to the cone points
        for i in range(len(x)):
            for j in range(len(x[i])):
                vec = np.array([x[i][j], y[i][j], z[i][j]])
                vec -= apex  # Translate to the origin
                if np.linalg.norm(axis_of_rotation) != 0:
                    rotated_vec = rotate_vector(vec, axis_of_rotation, angle_of_rotation)
                else:
                    rotated_vec = vec  # No rotation needed if the axis is zero
                rotated_vec += apex  # Translate back
                x[i][j], y[i][j], z[i][j] = rotated_vec

        xx, yy = np.meshgrid(np.linspace(-75, 75, 75), np.linspace(-75, 75, 75))

        d = np.dot(direction_vector, apex)
        zz = (d - direction_vector[0] * xx - direction_vector[1] * yy) / direction_vector[2]

        # ax.plot_surface(xx, yy, zz, color='yellow', alpha=0.5, label="Plane")
        for point in points: 
            vec = origin - apex
            vec_norm = vec / np.linalg.norm(vec)
            cone_dist = np.dot(origin - point, vec_norm)
            cone_radius = (cone_dist / distance) * radius
            orth_distance = np.linalg.norm((origin - point) - cone_dist * vec_norm)
            valid = np.dot(point, origin)
            if orth_distance <= cone_radius and (valid > 0):
                obs_POIs.append(point)
            else:
                non_obs.append(point)

        # ax.plot_wireframe(x, y, z, cmap=cm.coolwarm, alpha=0.3, label="FOV Cone of Spacecraft")
        ax.plot([origin[0], apex[0]], [origin[1], apex[1]], [origin[2], apex[2]], color='red', label='Axis of the FOV Cone', linewidth=5)
        ax.scatter(origin[0], origin[1], origin[2], marker='o', color='blue', s=100, label='Spacecraft Position', linewidth=5)

    non_obs = np.array(non_obs)
    obs_POIs = np.array(obs_POIs)
    n_obs = []
    # print(len(non_obs), len(obs_POIs))
    for p in non_obs:
        # check if the point is in obs_POIs
        if (p not in obs_POIs):
            n_obs.append(p)

    # remove all repeated points in obs_POIs
    obs_POIs = np.unique(obs_POIs, axis=0)
    print(len(obs_POIs) / 5000)
ax.scatter(obs_POIs[:, 0], obs_POIs[:, 1], obs_POIs[:, 2], color='green', label='POIs in View', s=1)
n_obs = np.array(n_obs)
ax.scatter(n_obs[:, 0], n_obs[:, 1], n_obs[:, 2], color='magenta', label='POIs not in View', s=1)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])
ax.scatter(apex[0], apex[1], apex[2], color='red', s=100, label='Expected Position of ISO')

# Ensure all labels are added and handle legend items explicitly:
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys())

plt.show()
