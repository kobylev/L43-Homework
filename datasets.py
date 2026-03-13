def get_dataloaders(noise_factor=config.TRAIN_NOISE_FACTOR):
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
