import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import config

class AddGaussianNoise(object):
    """
    Adds Gaussian noise to a tensor.
    The noise level is dynamically scaled based on the max pixel value.
    The requirements state a noise level between 0.01% to 10% of the pixel value.
    For simplicity and following standard practices, we will use a normal distribution 
    (mean 0, variance 1) multiplied by a `noise_factor`. 
    If a tensor is normalized between 0-1, `noise_factor` acts as the max percentage.
    """
    def __init__(self, noise_factor=config.TRAIN_NOISE_FACTOR):
        self.noise_factor = noise_factor

    def __call__(self, tensor):
        # Generate Gaussian noise (mean 0, std 1)
        noise = torch.randn_like(tensor) * self.noise_factor
        # Add noise to image
        noisy_image = tensor + noise
        # Clip values to ensure they remain in the valid [0, 1] range after adding noise
        noisy_image = torch.clamp(noisy_image, 0., 1.)
        return noisy_image

    def __repr__(self):
        return self.__class__.__name__ + f'(noise_factor={self.noise_factor})'

def get_dataloaders(noise_factor=config.TRAIN_NOISE_FACTOR):
    """
    Returns data loaders for training and validation MNIST sets.
    The noisy loader returns a tuple of (noisy_image, clean_image) by ignoring the labels,
    because an Autoencoder's target is the clean original image itself.
    """
    # Base transforms: convert to tensor ([0, 1] range)
    # Note: MNIST is naturally 1 channel (grayscale)
    transform_clean = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Download and load the training data
    trainset = datasets.MNIST(root=config.DATA_DIR, train=True, download=True, transform=transform_clean)
    testset = datasets.MNIST(root=config.DATA_DIR, train=False, download=True, transform=transform_clean)

class AutoencoderCollate:
    def __init__(self, noise_factor):
        self.noise_adder = AddGaussianNoise(noise_factor=noise_factor)
        
    def __call__(self, batch):
        # batch is a list of tuples (image, label)
        # We discard labels
        clean_images = torch.stack([item[0] for item in batch])
        noisy_images = self.noise_adder(clean_images)
        return noisy_images, clean_images

def get_dataloaders(noise_factor=config.TRAIN_NOISE_FACTOR):
    """
    Returns data loaders for training and validation MNIST sets.
    The noisy loader returns a tuple of (noisy_image, clean_image) by ignoring the labels,
    because an Autoencoder's target is the clean original image itself.
    """
    # Base transforms: convert to tensor ([0, 1] range)
    # Note: MNIST is naturally 1 channel (grayscale)
    transform_clean = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Download and load the training data
    trainset = datasets.MNIST(root=config.DATA_DIR, train=True, download=True, transform=transform_clean)
    testset = datasets.MNIST(root=config.DATA_DIR, train=False, download=True, transform=transform_clean)

    collate_fn = AutoencoderCollate(noise_factor=noise_factor)

    train_loader = DataLoader(
        trainset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        # Set to 0 to avoid PyTorch Windows multiprocessing issues 
        # if the collate function still causes trouble, but 
        # using a top-level callable class usually fixes it.
        num_workers=2 if torch.cuda.is_available() else 0, 
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        testset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=2 if torch.cuda.is_available() else 0,
        collate_fn=collate_fn
    )

    return train_loader, test_loader
