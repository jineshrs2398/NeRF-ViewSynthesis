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
            #dimensions [height, width,3]

    return(ray_origins, ray_directions)

def render_flat_rays(ray_origins, ray_directions, near, far, num_samples, rand=False):
    t_vals = tf.linspace(near,far, num_samples)
    if rand:
        shape = list(ray_origins.shape[:-1]) + [num_samples]
        noise = tf.random.uniform(shape=shape) * (far - near) / num_samples
        t_vals = t_vals + noise

    # Equation: r(t) = o + td -> Building the "r" here.
    rays = ray_origins[..., None, :] + (
        ray_directions[..., None, :] * t_vals[..., None]
    )
    rays_flat = tf.reshape(rays, [-1, 3])
    rays_flat = encode_position(rays_flat)
    return (rays_flat, t_vals)

def map_fn(pose):
    # Maps individual pose to flattened rays and sample points
    (ray_origins, ray_directions) = get_rays(height=H, width=W,
                                             focal=focal, pose=pose)
    (rays_flat, t_vals) = render_flat_rays(ray_origins=ray_origins,
                            ray_directions=ray_directions,
                            near=2.0,
                            far=6.0,
                            num_samples=NUM_SAMPLES,
                            rand=True)
    
    return (rays_flat, t_vals)

#training split
split_index= int(num_images*0.8)

train_images = images[:split_index]
val_images = images[split_index:]

train_poses = poses[:split_index]
val_poses = poses[split_index:]

#Pipeline for training
train_img_ds = tf.data.Dataset.from_tensor_slices(train_images)
train_pose_ds = tf.data.Dataset.from_tensor_slices(train_poses)
train_ray_ds = train_pose_ds.map(map_fn, num_parallel_calls=AUTO)
 #From pose getting rays
training_ds = tf.data.Dataset.zip((train_img_ds, train_ray_ds))
train_ds = (
    training_ds.shuffle(BATCH_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True, num_parallel_calls = AUTO)
    .prefetch(AUTO)
)

#Pipeline for validation 
val_img_ds = tf.data.Dataset.from_tensor_slices(val_images)
val_pose_ds = tf.data.Dataset.from_tensor_slices(val_poses)
val_ray_ds = val_pose_ds.map(map_fn, num_parallel_calls = AUTO)
validation_ds = tf.data.Dataset.zip((val_img_ds, val_ray_ds))
val_ds = (
    validation_ds.shuffle(BATCH_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True, num_parallel_calls=AUTO)
    .prefetch(AUTO)
)

def get_nerf_model(num_layers, num_pos):
    inputs = tf.keras.Input(shape=(num_pos, 2*3*POS_ENCODE_DIMS +3))
    x = inputs
    for i in range(num_layers):
        x = tf.keras.layers.Dense(units=64, activation="relu")(x)
        if i % 4 == 0 and i > 0:
            x = tf.keras.layers.concatenate([x, inputs], axis=-1)

    outputs = tf.keras.layers.Dense(units=4)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)