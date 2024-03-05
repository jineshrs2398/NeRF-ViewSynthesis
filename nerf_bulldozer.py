import tensorflow as tf
import os
import glob
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

#initialize global variables
AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 5 
NUM_SAMPLES = 32
POS_ENCODE_DIMS = 16
EPOCHS = 40

url = (
    "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz"
)

data = tf.keras.utils.get_file(origin=url)

data = np.load(data) 
images = data["images"]
(num_images, H, W, _) = images.shape
(poses, focal)= (data["poses"], data["images"])

plt.imshow(images[np.random.randint(low=0, high=num_images)])
plt.show()

def encode_position(x):
    # Generate fourier tensor of the position
    positions = [x]
    for i in range(POS_ENCODE_DIMS):
        for fn in [tf.sin, tf.cos]:
            positions.append[fn(2.0**i*x)]

    return tf.concat(positions, axis=-1)

def get_rays(height, width, focal, pose):
    # Get origin point and direction vector of rays
    
    #meshgrid for rays
    i, j = tf.meshgrid(tf.range(width, dtype = tf.float32) , 
                       tf.range(height, dtype = tf.float32), 
                       indexing="xy")  # dimensions [height, width]
    
    #Normalized x coordintae --> pushing x to center
    transformed_i = (i - width/2) / focal
    transformed_j = (j - width/2) / focal

    directions = tf.stack([transformed_i, -transformed_j, tf.ones_like(i)], axis =-1)
                    # dimensions [height, width, 3]
    camera_matrix = pose[:3, :3]  #dimensions [3, 3]
    height_width_focal = pose[:3, -1]

    transformed_dirs = directions[..., None, :] #dimensions [height, width,1 , 3]
    camera_dirs = transformed_dirs * camera_matrix #dimensions [height, width,1 , 3]
    ray_directions = tf.reduce_sum(camera_dirs, axis=-1) #dimensions [height, width,1]
    ray_origins = tf.broadcast_to(height_width_focal, tf.shape(ray_directions))

    return(ray_origins, ray_directions)

def render_flat_rays(ray_origins, ray_directions, near, far, num_samples, rand=False):
    t_vals = tf.linspace(near,far, num_samples)
    if rand:
        shape = list(ray_origins.shape[:-1]) + [num_samples]
        noise = tf.random.uniform(shape=shape) * (far - near) / num_samples
        t_vals = t_vals + noise

    rays = ray_origins[..., ]