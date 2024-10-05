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
import random

# Seed for reproducibility
# np.random.seed(42)

# Ellipsoid radii
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


# Define the objective function to minimize
def objective_function(x):
    fov = np.pi/4
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
        
        #arr=[]
        direction_vector = origin - apex
        vec = origin - apex
        direction_vector = direction_vector / np.linalg.norm(direction_vector)

        xx, yy = np.meshgrid(np.linspace(-100, 100, 100), np.linspace(-100, 100, 100))

        d = np.dot(direction_vector, apex)
        zz = (d - direction_vector[0] * xx - direction_vector[1] * yy) / direction_vector[2]
        
        for point in points: 
            if np.dot(point, origin)>0:
                vec = origin - apex
                vec_norm = vec/((vec[0]**2+vec[1]**2+vec[2]**2)**0.5)
                cone_dist = np.dot(origin-point, vec_norm)
                cone_radius = (cone_dist / distance) * 50
                orth_distance = (((origin-point)-cone_dist*vec_norm)[0]**2+((origin-point)-cone_dist*vec_norm)[1]**2+((origin-point)-cone_dist*vec_norm)[2]**2)**0.5
                orth_dist = orth_dist+orth_distance
        

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
                #total_overlap= total_overlap+np.cos(init_vals[i] / init_vals[j])
    

    cost= -(orth_distance - total_overlap)*(erf(sqrt(2)*mu1*(mu1 - 1)/2)*erf(sqrt(2)*mu2*(mu2 - 1)/2)*erf(sqrt(2)*mu3*(mu3 - 1)/2) + erf(sqrt(2)*mu1*(mu1 - 1)/2)*erf(sqrt(2)*mu2*(mu2 - 1)/2)*erf(sqrt(2)*mu3*(mu3 + 1)/2) + erf(sqrt(2)*mu1*(mu1 - 1)/2)*erf(sqrt(2)*mu2*(mu2 + 1)/2)*erf(sqrt(2)*mu3*(mu3 - 1)/2) + erf(sqrt(2)*mu1*(mu1 - 1)/2)*erf(sqrt(2)*mu2*(mu2 + 1)/2)*erf(sqrt(2)*mu3*(mu3 + 1)/2) + erf(sqrt(2)*mu1*(mu1 + 1)/2)*erf(sqrt(2)*mu2*(mu2 - 1)/2)*erf(sqrt(2)*mu3*(mu3 - 1)/2) + erf(sqrt(2)*mu1*(mu1 + 1)/2)*erf(sqrt(2)*mu2*(mu2 - 1)/2)*erf(sqrt(2)*mu3*(mu3 + 1)/2) + erf(sqrt(2)*mu1*(mu1 + 1)/2)*erf(sqrt(2)*mu2*(mu2 + 1)/2)*erf(sqrt(2)*mu3*(mu3 - 1)/2) + erf(sqrt(2)*mu1*(mu1 + 1)/2)*erf(sqrt(2)*mu2*(mu2 + 1)/2)*erf(sqrt(2)*mu3*(mu3 + 1)/2))/8
    return cost

res = []
res_cost = []
x = [(np.random.randint(100, 600) * random.choice([-1, 1])), (np.random.randint(100, 600) * random.choice([-1, 1])), (np.random.randint(100, 600) * random.choice([-1, 1])), (np.random.uniform(0, 2*np.pi))] * 3
bounds = [(-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (0, 2 * np.pi)] * 3
result = minimize(objective_function, x, method = "Nelder-Mead", bounds = bounds)
res.append(result)
res.append(result.fun)

print("Objective function value", result.fun)
print(result.x)

