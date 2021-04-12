import os
import shutil
from dataset import FaceInTheWild
from model import Discriminator, Generator, DCGAN
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# define the device to use
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

save_dir = os.path.join(os.curdir, 'saved_models')
checkpoint_path = os.path.join(save_dir, 'checkpoint.pth')
trained_model_path = os.path.join(save_dir, 'trained_model.pth')

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

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
    for idx, image in enumerate(train_dataloader):
        if idx + 1 > n_images**2:
            break
        plt.subplot(n_images, n_images, idx + 1)
        image = image[0].permute(1, 2, 0)
        plt.imshow(image)
        plt.axis('off')
    plt.show()
    plt.close()

def format_time(start, end):
    """
    Computes the interval time between a start and an end point.
    :param start: starting time
    :param end: ending time
    :return:
    """
    elapsed_time = end - start
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_secs, elapsed_mins

# get the names of the people in the dataset.
people_names = get_people_names()
# use those names to access each directory and retrieve the images.
# since we are implementing a GAN model, we don't care about the people's name
# we only want to retrieve the image.
data = get_images_label(people_names)
# create the dataset
train_data = FaceInTheWild(data, 'train')
val_data = FaceInTheWild(data, 'val')

## building the model
latent_vector = torch.randn((100, 1))
input_size = latent_vector.shape[0]
generator = Generator(input_size).to(device)
discriminator = Discriminator().to(device)
dcgan = DCGAN(generator, discriminator, input_size).to(device)

# hyperparameters
batch_size = 64
optim_params = {'lr': 0.0002, 'betas': (0.5, 0.999)}
n_epochs = 10
# dataloaders
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

real_img = 1
fake_img = 0


# train loop
def load_from_checkpoint():
    epoch_idx = 0
    d_loss_history = []
    g_loss_history = []
    generated_images = []
    if os.path.exists(checkpoint_path):
        loaded_checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        print('Checkpoint found. Restore from [{}/{}] epoch.'.format(loaded_checkpoint['epoch'], n_epochs))
        epoch_idx = loaded_checkpoint['epoch']
        dcgan.load_state_dict(loaded_checkpoint['dcgan_model'])
        dcgan.discriminator.optim.load_state_dict(loaded_checkpoint['d_optim'])
        dcgan.generator.optim.load_state_dict(loaded_checkpoint['g_optim'])
        d_loss_history = loaded_checkpoint['d_loss_history']
        g_loss_history = loaded_checkpoint['g_loss_history']
        generated_images = loaded_checkpoint['sample_imgs']
        return epoch_idx, d_loss_history, g_loss_history, generated_images
    else:
        return epoch_idx, d_loss_history, g_loss_history, generated_images

def save_trained_model(state_dict):
    print('Saving trained model...')
    torch.save(state_dict, trained_model_path)
    print('Model saved correctly.')


def train_loop():
    epoch_idx, d_loss_history, g_loss_history, sample_gen_images = load_from_checkpoint()
    for epoch in range(epoch_idx, n_epochs):
        d_loss_running = []
        g_loss_running = []
        for idx, batch in enumerate(train_dataloader):
            # TRAIN PROCEDURE
            # get batch of real images
            # each image has a shape of [NxCxHxW] --> [64x3x64x64]
            d_loss, g_loss, D_X, D_G_z = dcgan(batch)
            # store losses
            d_loss_running.append(d_loss)
            g_loss_running.append(g_loss)

            if idx % 25 == 0:
                print('Batch [{}/{}], D_Loss: {}, G_Loss: {}, D(x): {}, D(G(z)): {}'.format(idx,
                                                                                               train_dataloader.__len__(),
                                                                                               d_loss_running[-1],
                                                                                               d_loss_running[-1],
                                                                                               D_X,
                                                                                               D_G_z))
        d_loss_history.append(np.mean(d_loss_running).item())
        g_loss_history.append(np.mean(g_loss_running).item())
        checkpoint = {'epoch': epoch,
                      'dcgan_model': dcgan.state_dict(),
                      'd_optim': dcgan.discriminator.optim.state_dict(),
                      'g_optim': dcgan.generator.optim.state_dict(),
                      'd_loss_history': d_loss_history,
                      'g_loss_history': g_loss_history}
        if epoch % 2 == 0:
            # visualize a sample of generated images once every n epoch
            gen_images = dcgan.apply_knowledge()
            sample_gen_images.append(gen_images)
            dcgan.visualize_sample(gen_images)
            checkpoint['sample_imgs'] = sample_gen_images
        torch.save(checkpoint, checkpoint_path)
    print('Training Complete.')
    save_trained_model(dcgan.state_dict())

train_loop()






