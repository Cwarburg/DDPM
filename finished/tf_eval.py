import tensorflow as tf
import tensorflow_hub as tfhub
import tensorflow_gan as tfgan
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils import data
from PIL import Image
from tqdm import tqdm
import logging
from pathlib import Path
import sys
import os



MNIST_MODULE = "https://tfhub.dev/tensorflow/tfgan/eval/mnist/logits/1"
mnist_classifier_fn = tfhub.load(MNIST_MODULE)

def pack_images_to_tensor(path, model, img_size=None):
    """
    Given a path, pack all images into a tensor of shape (nb_images, height, width, channels)
    """
    nb_images = len(list(path.rglob("*.png")))
    #logger.info(f"Computing statistics for {nb_images} images")
    if model.lower() == "mnist":
        images = np.empty((nb_images, 28, 28, 1))  # TODO: Consider the RGB case
    else:
        images = np.empty((nb_images, 32, 32, 3))
    for idx, f in enumerate(tqdm(path.rglob("*.png"))):
        img = Image.open(f).convert("L") if model.lower() == "mnist" else Image.open(f)
        # resize if not the right size
        if img_size is not None and img.size[:2] != img_size:
            img = img.resize(
                size=(img_size[0], img_size[1]),
                resample=Image.BILINEAR,
            )
        img = np.array(img) / 255.0
        images[idx] = img[..., None] if model.lower() == "mnist" else img
    images_tf = tf.convert_to_tensor(images, dtype=tf.float32)
    return images_tf

def load_mnist():
    ds = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(28),
                transforms.ToTensor(),
            ]
        ),
    )
    dl = data.DataLoader(ds, batch_size=60000, shuffle=False)
    x, y = next(iter(dl))
    x = torch.permute(x, (0, 2, 3, 1))
    return tf.convert_to_tensor(x.numpy())

def load_cifar10():
    ds = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(32),
                transforms.ToTensor(),
            ]
        ),
    )
    dl = data.DataLoader(ds, batch_size=60000, shuffle=False)
    x, y = next(iter(dl))
    x = torch.permute(x, (0, 2, 3, 1))
    return tf.convert_to_tensor(x.numpy())

def compute_activations(tensors, num_batches, classifier_fn):
    """
    Given a tensor of of shape (batch_size, height, width, channels), computes
    the activiations given by classifier_fn.
    """
    tensors_list = tf.split(tensors, num_or_size_splits=num_batches)
    stack = tf.stack(tensors_list)
    activation = tf.nest.map_structure(
        tf.stop_gradient,
        tf.map_fn(classifier_fn, stack, parallel_iterations=1, swap_memory=True),
    )
    return tf.concat(tf.unstack(activation), 0)

def compute_real_stats(model, classifier_fn):
    if model.lower() == "mnist":
        data = load_mnist()
    elif model.lower() == "cifar10":
        data = load_cifar10()
    num_batches = 1
    activations1 = compute_activations(data, num_batches, classifier_fn)
    return activations1

def save_activations(activations, path):
    np.save(path, activations.numpy())


def main():

    print("Loading activations")

    model = sys.argv[1]
    if model.lower() == "mnist":
        classifier_fn = mnist_classifier_fn
    elif model.lower() == "cifar10":
        raise NotImplementedError
        #classifier_fn = cifar10_classifier_fn

    activations_real = np.load(f"STATS/ACTIVATIONS/{model.lower()}_activations.npy")
    activations_real = tf.convert_to_tensor(activations_real, dtype=tf.float32) 
    
    data_path = sys.argv[2]
    subfolders = [ f.path for f in os.scandir(data_path) if f.is_dir() ]
    epoch_dirs = sorted(subfolders, key=lambda x: int(x.split("_")[-1]))

    fid_scores = np.zeros(len(epoch_dirs))

    print("Calculating FID scores")
    for i, epoch_dir in enumerate(epoch_dirs):
        epoch_dir = Path(epoch_dir)

        epoch_images = pack_images_to_tensor(
            path=epoch_dir,
            )
        activation_fake = compute_activations(epoch_images, 1, classifier_fn)        
        fid = tfgan.eval.frechet_classifier_distance_from_activations(
            activations_real, activation_fake
            )

        fid_scores[i] = fid

        if i % 10 == 0:
            print(f"Epoch {i} FID: {fid}")

    print(f"Final FID score: {fid}")

    save_path = f"STATS/fid_scores_{model.lower()}.npy"
    
    np.save(save_path, fid_scores)


if __name__ == "__main__":
    main()
