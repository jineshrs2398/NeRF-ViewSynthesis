import tensorflow as tf
import os
import glob
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow import keras

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

def render_rgb_depth(model, rays_flat, t_vals, rand=True, train=True):
    # Generate RGB image and depth map from model predictions

    if train:
        predictions = model(rays_flat)
    else:
        predictions = model.predict(rays_flat)
    predictions = tf.reshape(predictions, shape=(BATCH_SIZE, H, W, NUM_SAMPLES, 4))

    rgb = tf.sigmoid(predictions[..., :-1])
    sigma_a = tf.nn.relu(predictions[..., -1])

    delta = t_vals[..., 1:] - t_vals[..., :-1]
    if rand:
        delta = tf.concat(
            [delta, tf.broadcast_to([1e10], shape=(BATCH_SIZE, H, W, 1))], axis=-1
        )  
        alpha = 1.0 - tf.exp(-sigma_a * delta)
    else:
        delta = tf.concat(
            [delta, tf.broadcast_to([1e10], shape=(BATCH_SIZE, H, W, 1))], axis=-1
        )
        alpha = 1.0 - tf.exp(-sigma_a * delta[:, None, None, :])

    #Get Transmittance
    exp_term = 1.0 - alpha 
    epsilon = 1e-10
    transmittance = tf.math.cumprod(exp_term + epsilon, axis=-1, exclusive=True)
    weights = alpha * transmittance
    rgb = tf.reduce_sum(weights[..., None] * rgb, axis=-2)

    if rand:
        depth_map = tf.reduce_sum(weights * t_vals, axis=-1)
    else:
        depth_map = tf.reduce_sum(weights * t_vals[:, None, None], axis=-1)
    return (rgb, depth_map)

class NeRF(keras.Model):
    def __init__(self, nerf_model):
        super().__init__()
        self.nerf_model = nerf_model

    def compile(self, optimizer, loss_fn):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.psnr_metric = keras.metrics.Mean(name="psnr")

    def train_step(self, inputs):
        # Get the images and the rays
        (images, rays) = inputs
        (rays_flat, t_vals) = rays

        with tf.GradientTape() as tape:
            #Get the predictions from the model
            rgb, _ = render_rgb_depth(
                model=self.nerf_model, rays_flat=rays_flat, t_vals=t_vals, rand=True
            )
            loss = self.loss_fn(images,rgb)

        # Get the trainable variables    
        trainable_variables = self.nerf_model.trainable_variables

        # Gradiant of the trainable variables with respect to the loss
        gradients = tape.gradient(loss, trainable_variables)

        # Apply the grads and optimize the model
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        # Get the PSNR of the reconstructed images and the source images.
        psnr = tf.image.psnr(images, rgb, max_val=1.0)

        #Compute metrics
        self.loss_tracker.update_state(loss)
        self.psnr_metric.update_state(psnr)
        return{"loss": self.loss_tracker.result(), "psnr": self.psnr_metric.result()}
    
    @property
    def metrics(self):
        return [self.loss_tracker, self.psnr_metric]
    
test_imgs, test_rays = next(iter(train_ds))
test_rays_flat, test_t_vals = test_rays

loss_list = []

class TrainMonitor(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        loss = logs["loss"]
        loss_list.append(loss)
        test_recons_images, depth_maps = render_rgb_depth(
            model=self.model.nerf_model,
            rays_flat=test_rays_flat,
            t_vals=test_t_vals,
            rand=True,
            train=False,
        )

        # Plot the rgb, depth and the loss plot
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20,5))
        ax[0].imshow(keras.utils.array_to_img)(test_recons_images[0])
        ax[0].set_title(f"Predicted Image: {epoch:03d}")

        ax[1].imshow(keras.utils.array_to_img(depth_maps[0, ..., None]))
        ax[1].set_title(f"Depth Map: {epoch:03d}")

        ax[2].plot(loss_list)
        ax[2].set_xticks(np.arange(0, EPOCHS + 1, 5.0))
        ax[2].set_title(f"Loss Plot: {epoch:03d}")

        fig.savefig(f"images/{epoch:03d}.png")
        plt.show()
        plt.close()

num_pos = H * W *NUM_SAMPLES
nerf_model = get_nerf_model(num_layers=8, num_pos=num_pos)

model = NeRF(nerf_model=nerf_model)
model.compile(
    optimizer=keras.optimizers.Adam(), loss_fn=keras.losses.MeanSquaredError()
)

# Create a dictionary to save the images during training
if not os.path.exists("images"):
    os.makedirs("images")

model.fit(
    train_ds,
    validation_dat = val_ds,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[TrainMonitor()],
)

def create_gif(path_to_images, name_gif):
    filenames = glob.glob(path_to_images)
    filenames = sorted(filenames)
    images = []
    for filename in tqdm(filenames):
        images.append(imageio.imread(filename))
    kargs = {"duration": 0.25}
    imageio.mimsave(name_gif, images, "GIF", **kargs)

create_gif("images/*.png", "training.gif")


# Get the trained NeRF model and infer
nerf_model = model.nerf_model
test_recons_images, depth_maps = render_rgb_depth(
    model=nerf_model,
    rays_flat=test_rays_flat,
    t_vals=test_t_vals,
    rand=True,
    train=False,
)

# Creating Sub Plots
figs, axes = plt.subplots(nrows=5, ncols=3, figsize=(10,20))

for ax, ori_img, recons_img, depth_map in zip(
    axes, test_imgs, test_recons_images, depth_maps
):
    ax[0].imshow(keras.utils.array_to_img(ori_img))
    ax[0].set_title("Original")

    ax[1].imshow(keras.utils.array_to_img(recons_img))
    ax[1].set_title("Reconstructed")

    ax[2].imshow(keras.utils.array_to_img(depth_map[..., None]), cmap="inferno")
    ax[2].set_title("Depth Map")


# Rendering 3D Scenes
def get_translation_t(t):
    # Translation matrix for movement in t
    matrix = [
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,t],
        [0,0,0,1],
        ]
    return tf.convert_to_tensor(matrix, dtype=tf.float32)

def get_rotation_phi(phi):
    # Rotation matrix for movement in phi
    matrix = [
        [1,0,0,0],
        [0, tf.cos(phi), -tf.sin(phi), 0],
        [0, tf.sin(phi), tf.cos(phi), 0],
        [0,0,0,1],
    ]
    return tf.convert_to_tensor(matrix, dtype=tf.float32)

def get_rotation_theta(theta):
    # ROtation matrix for movement in theta
    matrix = [
        [tf.cos(theta), 0, -tf.sin(theta), 0],
        [0, 1, 0, 0],
        [tf.sin(theta), 0, tf.cos(theta), 0],
        [0, 0, 0, 1],
    ]
    return tf.convert_to_tensor(matrix, dtype=tf.float32)

def pose_spherical(theta, phi, t):
    # Camera to world matix for theta, phi and t
    c2w = get_translation_t(t)
    c2w = get_rotation_phi(phi/180.0 * np.pi) @c2w
    c2w = get_rotation_theta(theta/180.0 * np.pi) @c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @c2w
    return c2w 

rgb_frames = []
batch_flat = []
batch_t = []

for index, theta in tqdm(enumerate(np.linspace(0.0, 360.0, 120, endpoint=False))):
    c2w = pose_spherical(theta, -30.0, 4.0)

    ray_oris, ray_dirs = get_rays(H, W, focal, c2w)
    rays_flats, t_vals = render_flat_rays(
        ray_oris, ray_dirs, near=2.0, far=6.0, num_samples=NUM_SAMPLES, rand=False
    )

    if index % BATCH_SIZE == 0 and index > 0:
        batched_flat = tf.stack(batch_flat, axis=0)
        batch_flat = [rays_flats]

        batched_t = tf.stack(batch_t, axis=0)
        batch_t = [t_vals]

        rgb,_ = render_rgb_depth(
            nerf_model, batched_flat, batch_t, rand=False, train=False
        )

        temp_rgb = [np.clip(255*img, 0.0, 255.0).astype(np.uint8) for img in rgb]
        rgb_frames = rgb_frames + temp_rgb

    else:
        batch_flat.append(rays_flats)
        batch_t.append(t_vals)

rgb_video = "rgb_video.mp4"
imageio.miwrite(rgb_video, rgb_frames, fps=30, quality=7, macro_block_size=None)