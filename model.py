import torch
import torch.nn as nn

class DenoisingAutoencoder(nn.Module):
    """
    A simple Convolutional Autoencoder designed to denoise 28x28 grayscale MNIST images.
    """
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        
        # Encoder
        # Input size: [batch_size, 1, 28, 28]
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # -> [16, 14, 14]
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # -> [32, 7, 7]
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=7)                       # -> [64, 1, 1]
        )
        
        # Decoder
        # Input size: [batch_size, 64, 1, 1]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=7),             # -> [32, 7, 7]
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # -> [16, 14, 14]
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> [1, 28, 28]
            nn.Sigmoid() # Use Sigmoid to ensure pixel values are bound between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

if __name__ == '__main__':
    # Test model input/output dimensions
    model = DenoisingAutoencoder()
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert dummy_input.shape == output.shape, "Output dimensions must match input dimensions!"
