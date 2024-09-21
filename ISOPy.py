import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import scipy
from scipy.optimize import minimize
from scipy.special import erf

from numpy import sqrt
from tqdm import tqdm 
import itertools
from numpy import cos 
from numpy import tan


covariance= np.array([[2677226.887245484, 285.21446641916737, 268.99258277423775, 0.4395130785408834, 0.04869538309075001, 0.04977713754424471], 
               [285.21446641916737, 0.9848154379616082, 0.9539977104619806, 5.143220640656547e-05, 2.402194457343449e-05, 1.1448019563650868e-05], 
               [268.99258277423775, 0.9539977104619806, 0.9362931818523705, 4.7953059657441394e-05, 2.0087537013333486e-05, 1.0905119663080185e-05], 
               [0.4395130785408834, 5.143220640656547e-05, 4.7953059657441394e-05, 9.116068280599878e-08, 9.657403280723735e-09, 9.569364053017187e-09], 
               [0.04869538309075001, 2.402194457343449e-05, 2.0087537013333486e-05, 9.657403280723735e-09, 2.632076219713602e-09, 1.587031182752267e-09], 
               [0.04977713754424471, 1.1448019563650868e-05, 1.0905119663080185e-05, 9.569364053017187e-09, 1.587031182752267e-09, 1.6117378623661833e-09]])

position_covariances= covariance[:3, :3] 
# since the covariance matrix is a 6x6 matrix for each state, position_covariances extracts a 3x3 
# matrix for the covariances relating the x, y, and z positions to each other 

# the radius in the x, y, and z directions of the uncertainty sphere is calculated using the means of each row of the position_covariances 
# matrix. Because the x positions are much larger than the y and z, the x radius was scaled by 1e-4 so it would be on the same magnitude as 
# the y and z radii

x_rad = np.mean(position_covariances[0]) * 1e-4
y_rad = np.mean(position_covariances[1])
z_rad = np.mean(position_covariances[2])

x1, x2, x3, x4, x5, x6, mu1, sigma1, mu2, sigma2, mu3, sigma3, mu4, sigma4, mu5, sigma5, mu6, sigma6 = sym.symbols('x1 x2 x3 x4 x5 x6 mu1 sigma1 mu2 sigma2 mu3 sigma3 mu4 sigma4 mu5 sigma5 mu6 sigma6')

theta, phi= sym.symbols('theta phi')
total_overlap, orth_distance= sym.symbols('total_overlap, orth_distance')

def normal_pdf(x, mu, sigma):
    return (1 / (sym.sqrt(2 * sym.pi) * sigma)) * sym.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def info_cost(mu1, mu2, mu3, theta, phi):
    cost= total_overlap-orth_distance
    return cost

integrand = normal_pdf(x1, mu1, sigma1) * normal_pdf(x2, mu2, sigma2)*normal_pdf(x3, mu3, sigma3) * info_cost(mu1, mu2, mu3, theta, phi)

integral_expr = sym.integrate(integrand, (x1, -(mu1**2*sigma1), (mu1**2*sigma1)), (x2,  -(mu2**2*sigma2), (mu2**2*sigma2)), (x3,  -(mu3**2*sigma3), (mu3**2*sigma3)))

#these are symbols used in the portion of the cost function for the second spacecraft (ignore for now)
mu1_2, mu2_2, mu3_2, mu4_2, mu5_2, mu6_2, theta_2, phi_2= sym.symbols('mu1_2, mu2_2, mu3_2, mu4_2, mu5_2, mu6_2, theta_2, phi_2')

result_expr = integral_expr.subs({
    sigma1: 1, sigma2: 1, sigma3: 1, sigma4: 1, sigma5: 1, sigma6: 1
}).simplify().simplify()

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
n = 8000
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

plt.show()

fov=np.pi/4

# Define the objective function to minimize
def objective_function(x):
    init_vals=x
    apex=np.array([0, 0, 0])
    pos = 0
    norm_pos = 0
    total_overlap = 0.0
    tot_vol=0 
    orth_dist=0


    for i in range(3, len(init_vals), 4):
        fov_range = [np.degrees(fov) -  init_vals[i]/ 2, np.degrees(fov) + init_vals[i] / 2]
        pos = np.array([init_vals[i-3], init_vals[i-2], init_vals[i-1]]) *1e2
        mu1=pos[0]
        mu2=pos[1]
        mu3=pos[2]
        origin=pos
        norm_pos = (pos[0]**2+pos[1]**2+pos[2]**2)**0.5
        distance= ((origin - apex)[0]**2+(origin - apex)[1]**2+(origin - apex)[2]**2)**0.5
        
        arr=[]
        for point in points: 
            vec = origin - apex
            vec_norm= vec/((vec[0]**2+vec[1]**2+vec[2]**2)**0.5)
            cone_dist = np.dot(origin-point, vec_norm)
            cone_radius = (cone_dist / distance) * 50
            orth_distance = (((origin-point)-cone_dist*vec_norm)[0]**2+((origin-point)-cone_dist*vec_norm)[1]**2+((origin-point)-cone_dist*vec_norm)[2]**2)**0.5
            
            # if orth_distance<=cone_radius:
            #     arr.append(point)
            #orth_dist= orth_dist+orth_distance

        for j in range(i + 4, len(init_vals), 4):
            start1, end1 = init_vals[i] - fov / 2, init_vals[i] + fov / 2
            start2, end2 = init_vals[j] - fov / 2, init_vals[j] + fov / 2
            overlap_start = max(start1, start2)
            overlap_end = min(end1, end2)
    
            overlap= max(0, overlap_end-overlap_start)
                
            total_overlap= total_overlap+overlap
                
            if np.abs(init_vals[i] - init_vals[j]) < 0.01:
                if init_vals[j]==0:
                    init_vals[j]= init_vals[j]+0.01
                total_overlap= total_overlap+np.cos(init_vals[i] / init_vals[j])
    #mu1, mu2, mu3, mu4, mu5, mu6 = x[:6]

    #the mu1 initial condition of the first SC was scaled by 1e-4 because the x pos. state is an order of 1e4 larger than the y and z positions and 
    #we do not want this to unnecessarily influence the optimization

    cost= -(orth_distance - total_overlap)*(erf(sqrt(2)*mu1*(mu1 - 1)/2)*erf(sqrt(2)*mu2*(mu2 - 1)/2)*erf(sqrt(2)*mu3*(mu3 - 1)/2) + erf(sqrt(2)*mu1*(mu1 - 1)/2)*erf(sqrt(2)*mu2*(mu2 - 1)/2)*erf(sqrt(2)*mu3*(mu3 + 1)/2) + erf(sqrt(2)*mu1*(mu1 - 1)/2)*erf(sqrt(2)*mu2*(mu2 + 1)/2)*erf(sqrt(2)*mu3*(mu3 - 1)/2) + erf(sqrt(2)*mu1*(mu1 - 1)/2)*erf(sqrt(2)*mu2*(mu2 + 1)/2)*erf(sqrt(2)*mu3*(mu3 + 1)/2) + erf(sqrt(2)*mu1*(mu1 + 1)/2)*erf(sqrt(2)*mu2*(mu2 - 1)/2)*erf(sqrt(2)*mu3*(mu3 - 1)/2) + erf(sqrt(2)*mu1*(mu1 + 1)/2)*erf(sqrt(2)*mu2*(mu2 - 1)/2)*erf(sqrt(2)*mu3*(mu3 + 1)/2) + erf(sqrt(2)*mu1*(mu1 + 1)/2)*erf(sqrt(2)*mu2*(mu2 + 1)/2)*erf(sqrt(2)*mu3*(mu3 - 1)/2) + erf(sqrt(2)*mu1*(mu1 + 1)/2)*erf(sqrt(2)*mu2*(mu2 + 1)/2)*erf(sqrt(2)*mu3*(mu3 + 1)/2))/8
    #cost 1 is the cost for spacecraft 1 and cost 2 is for spacecraft 2
    return cost
 

x = np.array([100, 200, 300, np.pi/8, 100, 200, 300, np.pi/8, 100, 200, 300, np.pi/8])

#all of the variables were left unbounded except for the angles. In the FOV equation, we see that AFOV is bounded by arctan, which has 
#bounds of -90 to 90 degrees. Phi was left as bounded by the standard angle range, 0-360 degrees.
bounds = [(-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (0, 2 * np.pi)] * 3

result= minimize(objective_function, x, method= "Nelder-Mead", bounds= bounds) #I chose to use the Nelder-Mead method of 
#optimizaition because it is robust to noisy functions, like this one 

print("Optimized distance of SC1", result.x[0], result.x[1], result.x[2])
print("Optimized angles of SC1 in degrees", np.rad2deg(result.x[3]))
print("Optimized distance of SC2", result.x[4], result.x[5], result.x[6])
print("Optimized angles of SC2 in degrees", np.rad2deg(result.x[7]))
print("Optimized distance of SC3", result.x[8], result.x[9], result.x[10])
print("Optimized angles of SC3 in degrees", np.rad2deg(result.x[11]))
print("Objective function value", result.fun)
print(result.x)

pos_1= result.x[:3]
pos_2=result.x[4:7]
pos_3= result.x[8:11]
#pos_4= result.x[12:15]
#pos_5= result.x[16:19]
#pos_5= result.x[21:24]
#np.degrees(result.x[-1])
angle_1= np.degrees(np.arccos(pos_1[2]/ np.linalg.norm(pos_1)))
angle_2= np.degrees(np.arccos(pos_2[2]/ np.linalg.norm(pos_2)))
angle_3= np.degrees(np.arccos(pos_3[2]/ np.linalg.norm(pos_3)))

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Function to rotate a vector around a given axis by a given angle
def rotate_vector(vector, axis, angle):
    axis = axis / np.linalg.norm(axis)
    vector = vector * np.cos(angle) + np.cross(axis, vector) * np.sin(angle) + axis * np.dot(axis, vector) * (1 - np.cos(angle))
    return vector

# Define the position of the apex of the cone (green X)
origin = np.array([pos_1[0], pos_1[1], pos_1[2]])

# Define the origin
apex = np.array([0, 0, 0])

# Create a figure
fig = plt.figure()
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

half_ang=np.arctan(50/distance)

# Translate the cone to have its apex at the specified position
x += apex[0]
y += apex[1]
z += apex[2]

# Calculate the direction vector from the apex to the origin
direction_vector = origin - apex
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
ax.quiver(apex[0], apex[1], apex[2], 
          direction_vector[0]*arrow_length, direction_vector[1]*arrow_length, direction_vector[2]*arrow_length, 
          color='r', label='Viewing Direction')

# Plot the FOV cone
ax.plot_surface(x, y, z, cmap=cm.coolwarm, alpha=0.7)

# Plot the sphere
r = 0.05
u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
x_rad, y_rad, z_rad = 89.25936980982259, 95.71775985586366, 90.29429122218403

x_sphere = x_rad * np.cos(u) * np.sin(v)
y_sphere = y_rad * np.sin(u) * np.sin(v)
z_sphere = z_rad * np.cos(v)
ax.plot_surface(x_sphere, y_sphere, z_sphere, cmap='rainbow', alpha=0.5)

# Add markers
ax.plot(0, 0, 0, color='green', marker='o')
#ax.plot(origin[0], origin[1], origin[2], color='green', marker='X', markersize= 10)

# Add axes
ax.quiver(0, 0, 0, x_rad, 0, 0, color='r', label='X-axis')
ax.quiver(0, 0, 0, 0, y_rad, 0, color='g', label='Y-axis')
ax.quiver(0, 0, 0, 0, 0, z_rad, color='b', label='Z-axis')

for point in points: 
    vec = origin - apex
    vec_norm= vec/np.linalg.norm(vec)
    cone_dist = np.dot(origin-point, vec_norm)
    cone_radius = (cone_dist / distance) * 50
    orth_distance = np.linalg.norm((origin-point)-cone_dist*vec_norm)
    if orth_distance<=cone_radius:
        ax.plot(point[0], point[1], point[2], color= 'green', marker= 'X')







# Define the position of the apex of the cone (green X)
origin = np.array([pos_2[0], pos_2[1], pos_2[2]])

# Define the origin
apex = np.array([0, 0, 0])


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

half_ang=np.arctan(50/distance)

# Translate the cone to have its apex at the specified position
x += apex[0]
y += apex[1]
z += apex[2]

# Calculate the direction vector from the apex to the origin
direction_vector = origin - apex
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
ax.quiver(apex[0], apex[1], apex[2], 
          direction_vector[0]*arrow_length, direction_vector[1]*arrow_length, direction_vector[2]*arrow_length, 
          color='r', label='Viewing Direction')

# Plot the FOV cone
ax.plot_surface(x, y, z, cmap=cm.coolwarm, alpha=0.7)

for point in points: 
    vec = origin - apex
    vec_norm= vec/np.linalg.norm(vec)
    cone_dist = np.dot(origin-point, vec_norm)
    cone_radius = (cone_dist / distance) * 50
    orth_distance = np.linalg.norm((origin-point)-cone_dist*vec_norm)
    if orth_distance<=cone_radius:
        ax.plot(point[0], point[1], point[2], color= 'blue', marker= 'X')






# Define the position of the apex of the cone (green X)
origin = np.array([pos_3[0], pos_3[1], pos_3[2]])

# Define the origin
apex = np.array([0, 0, 0])


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

half_ang=np.arctan(50/distance)

# Translate the cone to have its apex at the specified position
x += apex[0]
y += apex[1]
z += apex[2]

# Calculate the direction vector from the apex to the origin
direction_vector = origin - apex
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
ax.quiver(apex[0], apex[1], apex[2], 
          direction_vector[0]*arrow_length, direction_vector[1]*arrow_length, direction_vector[2]*arrow_length, 
          color='r', label='Viewing Direction')

# Plot the FOV cone
ax.plot_surface(x, y, z, cmap=cm.coolwarm, alpha=0.7)

for point in points: 
    vec = origin - apex
    vec_norm= vec/np.linalg.norm(vec)
    cone_dist = np.dot(origin-point, vec_norm)
    cone_radius = (cone_dist / distance) * 50
    orth_distance = np.linalg.norm((origin-point)-cone_dist*vec_norm)
    if orth_distance<=cone_radius:
        ax.plot(point[0], point[1], point[2], color= 'red', marker= 'X')




# # Define the position of the apex of the cone (green X)
# origin = np.array([pos_4[0], pos_4[1], pos_4[2]])

# # Define the origin
# apex = np.array([0, 0, 0])


# # Generate mesh grid for the cone
# u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi/2:80j]
# x = np.cos(u) * np.sin(v)
# y = np.sin(u) * np.sin(v)
# z = np.cos(v)  # Pointy part towards the z direction

# distance = np.linalg.norm(origin - apex)

# # Scale the cone to make it bigger
# scale_factor = 100  # Adjust this scale factor as needed
# x *= 50
# y *= 50
# z *= distance

# half_ang=np.arctan(50/distance)

# # Translate the cone to have its apex at the specified position
# x += apex[0]
# y += apex[1]
# z += apex[2]

# # Calculate the direction vector from the apex to the origin
# direction_vector = origin - apex
# direction_vector = direction_vector / np.linalg.norm(direction_vector)

# # Calculate the axis and angle of rotation to align the cone's axis with the direction vector
# axis_of_rotation = np.cross([0, 0, 1], direction_vector)
# angle_of_rotation = np.arccos(np.dot([0, 0, 1], direction_vector))

# # Apply rotation to the cone points
# for i in range(len(x)):
#     for j in range(len(x[i])):
#         vec = np.array([x[i][j], y[i][j], z[i][j]])
#         vec -= apex  # Translate to the origin
#         if np.linalg.norm(axis_of_rotation) != 0:
#             rotated_vec = rotate_vector(vec, axis_of_rotation, angle_of_rotation)
#         else:
#             rotated_vec = vec  # No rotation needed if the axis is zero
#         rotated_vec += apex  # Translate back
#         x[i][j], y[i][j], z[i][j] = rotated_vec

# # Plot the direction vector
# arrow_length = scale_factor * 0.5  # Adjust this to ensure the arrow is visible
# ax.quiver(apex[0], apex[1], apex[2], 
#           direction_vector[0]*arrow_length, direction_vector[1]*arrow_length, direction_vector[2]*arrow_length, 
#           color='r', label='Viewing Direction')

# # Plot the FOV cone
# #ax.plot_surface(x, y, z, cmap=cm.coolwarm, alpha=0.7)

# for point in points: 
#     vec = origin - apex
#     vec_norm= vec/np.linalg.norm(vec)
#     cone_dist = np.dot(origin-point, vec_norm)
#     cone_radius = (cone_dist / distance) * 50
#     orth_distance = np.linalg.norm((origin-point)-cone_dist*vec_norm)
#     if orth_distance<=cone_radius:
#         ax.plot(point[0], point[1], point[2], color= 'yellow', marker= 'X')



# Set labels and view
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
#ax.legend()

# Show the plot
plt.show()

