from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Seed for reproducibility
np.random.seed(42)

# Ellipsoid radii
x_rad, y_rad, z_rad = 89.25936980982259, 95.71775985586366, 90.29429122218403
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
n = 10000
points = []
while len(points) < n:
    x, y, z = randompoint()
    scaled_point = f(x, y, z)
    points.append(scaled_point)

# Convert list to numpy array for easier handling
points = np.array(points)

# Plot the generated points
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)

# Set aspect ratio for the axes
ax.set_box_aspect((np.ptp(points[:, 0]), np.ptp(points[:, 1]), np.ptp(points[:, 2])))

# Set labels for the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title("Random Sampling of 10000 POIs", fontsize=16)

plt.show()

x_rad, y_rad, z_rad = 89.25936980982259, 95.71775985586366, 90.29429122218403

# Function to rotate a vector around a given axis by a given angle
def rotate_vector(vector, axis, angle):
    axis = axis / np.linalg.norm(axis)
    vector = vector * np.cos(angle) + np.cross(axis, vector) * np.sin(angle) + axis * np.dot(axis, vector) * (1 - np.cos(angle))
    return vector

# the field of view (FOV)
fov = np.pi / 4

pos_1 = np.array([0, 150, 150])
# Define the position of the apex of the cone (green X)
origin = np.array([pos_1[0], pos_1[1], pos_1[2]])

# Define the origin
apex = np.array([0, 0, 0])

# Create a figure
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

# Generate mesh grid for the cone
u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi/2:80j]
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)  # Pointy part towards the z direction

distance = np.linalg.norm(origin - apex)

# Scale the cone to make it bigger
scale_factor = 100  # Adjust this scale factor as needed
x *= 50
y *= 50
z *= distance

half_ang = np.arctan(50 / distance)

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

# Plot the direction vector
arrow_length = scale_factor * 0.5  # Adjust this to ensure the arrow is visible
ax.quiver(origin[0], origin[1], origin[2], 
          -direction_vector[0] * distance, -direction_vector[1] * distance, -direction_vector[2] * distance, 
          color='black', label='Axis of FOV',  arrow_length_ratio=0.1)

# Plot the FOV cone
ax.plot_wireframe(x, y, z, cmap=cm.coolwarm, alpha=0.5, label="FOV Cone of Spacecraft")

# Plot the sphere
r = 0.05
u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]

x_sphere = x_rad * np.cos(u) * np.sin(v)
y_sphere = y_rad * np.sin(u) * np.sin(v)
z_sphere = z_rad * np.cos(v)
ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.2, label = "Uncertainty Ellipsoid")

ax.scatter(origin[0], origin[1], origin[2], color="red", label="$p_{sc}$", s=75, marker="x")

# Add axes
axes_rad = 20
ax.quiver(0, 0, 0, axes_rad, 0, 0, color='r')
ax.quiver(0, 0, 0, 0, axes_rad, 0, color='g')
ax.quiver(0, 0, 0, 0, 0, axes_rad, color='b')

ax.quiver(apex[0], apex[1], apex[2], direction_vector[0], direction_vector[1], direction_vector[2])

xx, yy = np.meshgrid(np.linspace(-100, 100, 100), np.linspace(-100, 100, 100))

d = np.dot(direction_vector, apex)
zz = (d - direction_vector[0] * xx - direction_vector[1] * yy) / direction_vector[2]

ax.plot_surface(xx, yy, zz, color='yellow', alpha=0.5, label="Plane")
obs_POIs = []
non_obs = []
for point in points: 
    vec = origin - apex
    vec_norm = vec / np.linalg.norm(vec)
    cone_dist = np.dot(origin - point, vec_norm)
    cone_radius = (cone_dist / distance) * 50
    orth_distance = np.linalg.norm((origin - point) - cone_dist * vec_norm)
    valid = np.dot(point, origin)
    if orth_distance <= cone_radius:
        if (valid > 0):
            obs_POIs.append(point)
        else:
            non_obs.append(point)
non_obs = np.array(non_obs)
obs_POIs = np.array(obs_POIs)
ax.scatter(obs_POIs[:,0], obs_POIs[:,1], obs_POIs[:,2], color= 'green', marker= 'X', label="POIs in View")
ax.scatter(non_obs[:,0], non_obs[:,1], non_obs[:,2], color= 'magenta', marker= 'X', label="POIs Not in View")
# Set labels and view
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
#ax.legend()

# Show the plot
plt.legend()
plt.show()
