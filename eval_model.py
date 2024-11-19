from pathlib import Path
import logging
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_gan as tfgan
import torch
from torchvision import datasets, transforms
import torch.utils.data as data
from tqdm import tqdm
epoch_dir = 'placeholder'
logger = logging.getLogger(__name__)

def pack_images_to_tensor(path, img_size=None):
    """
    Given a path, pack all images into a tensor of shape (nb_images, height, width, channels)
    """
    nb_images = len(list(path.rglob("*.png")))
    logger.info(f"Computing statistics for {nb_images} images")
    images = np.empty((nb_images, 28, 28, 1))  # TODO: Consider the RGB case
    for idx, f in enumerate(tqdm(path.rglob("*.png"))):
        img = Image.open(f)
        # resize if not the right size
        if img_size is not None and img.size[:2] != img_size:
            img = img.resize(
                size=(img_size[0], img_size[1]),
                resample=Image.BILINEAR,
            )
        img = np.array(img) / 255.0
        images[idx] = img[..., None]
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


def compute_mnist_stats(mnist_classifier_fn):
    mnist = load_mnist()
    num_batches = 1
    activations1 = compute_activations(mnist, num_batches, mnist_classifier_fn)
    return activations1


def save_activations(activations, path):
    np.save(path, activations.numpy())

activations_real = np.load("./data/mnist/activations_real.npy")
activations_real = tf.convert_to_tensor(activations_real, dtype=tf.float32)

def process_epoch(epoch_dir, activations_real, classifier_fn):
    logger.info(f"Loading images of epoch {epoch_dir.name}")
    epoch_images = pack_images_to_tensor(path=epoch_dir)
    logger.info("Computing fake activations")
    activation_fake = compute_activations(epoch_images, 1, classifier_fn)

    logger.info("Computing FID")
    fid = tfgan.eval.frechet_classifier_distance_from_activations(
        activations_real, activation_fake
    )
    logger.info(f"FID: {fid}")