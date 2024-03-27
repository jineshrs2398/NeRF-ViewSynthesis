# NeRF-ViewSynthesis

This implementation introduces a minimal version of NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis, as described in the [original NeRF paper](https://arxiv.org/abs/2003.08934) by Ben Mildenhall et al. It demonstrates how novel views of a scene can be synthesized by modeling the volumetric scene function through a neural network.
![Nerf GIF](https://github.com/jineshrs2398/NeRF-ViewSynthesis/blob/main/nerf.gif)

## Setup

The code requires TensorFlow and Keras for the implementation of the NeRF model. Ensure you have the latest version of TensorFlow installed to utilize GPU acceleration for training the model. Additional libraries include `numpy`, `matplotlib`, `imageio`, and `tqdm`.


## Dataset

The dataset used is the `tiny_nerf_data.npz` file, which contains images, camera poses, and focal length information. The data captures multiple views of a scene, enabling the neural network to learn a 3D representation of the scene.

## Model

The model is a Multi-Layer Perceptron (MLP) that takes encoded positions and viewing angles as input and outputs the RGB color and volume density at that point. This minimal implementation uses 64 Dense units per layer, as opposed to 256 as mentioned in the original paper.

## Training

To train the model, simply run:

```bash
python nerf_bulldozer.py
```
This script will automatically download the dataset, initiate training, and save the generated images during training to the `images/` directory. Training parameters such as the batch size, number of samples, and epochs are configured within the script.

## Inference

The training was run for 1000 epochs and the model can synthesize novel views of the scene by specifying different camera poses. The `render_rgb_depth` function generates RGB images and depth maps from the learned model, showcasing the model's ability to infer 3D scenes from a sparse set of 2D images.


## Reference and Further Reading

- [Original NeRF GitHub Repository](https://github.com/bmild/nerf)
- [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis Paper](https://arxiv.org/abs/2003.08934)
- [PyImageSearch NeRF Blog Series](https://www.pyimagesearch.com/2021/11/10/computer-graphics-and-deep-learning-with-nerf-using-tensorflow-and-keras-part-1/)

## License

This project is open-sourced under the MIT license.
