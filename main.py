import os
import shutil
from utils import FaceInTheWild
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils

##### PROCEDURE TO GET THE IMAGES #####
NAMES_DIR_PATH = os.path.join(os.curdir, 'dataset/lfw')


def get_people_names():
    """
    This function returns a list of the directories name of each person in the dataset.
    :return:
    """
    return os.listdir(NAMES_DIR_PATH)

def get_images_label(people):
    """
    This function iterates through an array of name's people to access the
    directory that contains the corresponding images and creates a new directory
    containing all the pictures of the people
    :param people: the list of the people names in the dataset.
    :return:
    """
    destination_path = os.path.join(os.curdir, 'images')
    if not os.path.exists(destination_path):
        os.mkdir('images')
    for person in people:
        path = os.path.join(NAMES_DIR_PATH, person)
        images = os.listdir(path)
        for image in images:
            image_path = os.path.join(path, image)
            if not os.path.exists(os.path.join(destination_path, image)):
                shutil.copy(image_path, destination_path)
    print('Images parsed correctly.')
    print('Total number of images: {}'.format(len(os.listdir(destination_path))))
    images = os.listdir(destination_path)
    return images

def visualize_sample_image(n_images):
    """
    This function visualize n_images, sampled randomly
    from the dataset
    :param n_images: number of image to print per axis
    :return:
    """
    plt.figure(figsize=(8, 8))
    for idx, image in enumerate(dataloader):
        if idx + 1 > n_images**2:
            break
        plt.subplot(n_images, n_images, idx + 1)
        image = image[0].permute(1, 2, 0)
        plt.imshow(image)
        plt.axis('off')
    plt.show()
    plt.close()

# get the names of the people in the dataset.
people_names = get_people_names()
# use those names to access each directory and retrieve the images.
# since we are implementing a GAN model, we don't care about the people's name
# we only want to retrieve the image.
data = get_images_label(people_names)
# create the dataset
dataset = FaceInTheWild(data, 'train')
# create the dataloader
kwargs = {'batch_size': 128, 'shuffle': True}
dataloader = DataLoader(dataset, **kwargs)

# define the device to use
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

# visualize some of the images
visualize_sample_image(6)



