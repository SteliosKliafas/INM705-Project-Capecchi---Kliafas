import torchvision
from PIL import Image
import os
from torch.utils.data import Dataset


class FaceInTheWild(Dataset):

    IMAGES_PATH_TRAIN = os.path.join(os.curdir, 'images/train')
    IMAGES_PATH_TEST = os.path.join(os.curdir, 'images/test')

    def __init__(self, data, phase):
        super(FaceInTheWild, self).__init__()
        self.data = data
        self.phase = phase

    def retrieve_image(self, image_label):
        """
        This function fetch the desired image from its directory and converts it to a torch tensor.
        :param image_label: the image label used to retrieve the image from the parent directory
        :return image: the tensor image of shape (3x64x64) -> (CxHxW)
        """
        if self.phase == 'train':
            image_path = os.path.join(FaceInTheWild.IMAGES_PATH_TRAIN, image_label)
        elif self.phase == 'test':
            image_path = os.path.join(FaceInTheWild.IMAGES_PATH_TEST, image_label)
        # by defaults, the Discriminator and Generator accepts an image of shape (3x64x64), hence we rescale.
        tensor_converter = torchvision.transforms.Compose([torchvision.transforms.Resize((64, 64)),
                                                           torchvision.transforms.ToTensor(),
                                                           torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                                                            (0.5, 0.5, 0.5))])
        image = tensor_converter(Image.open(image_path))
        return image

    def __getitem__(self, idx):
        image = self.retrieve_image(self.data[idx])
        return image

    def __len__(self):
        return len(self.data)
