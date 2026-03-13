import torch
import matplotlib.pyplot as plt
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import config
from model import DenoisingAutoencoder
from datasets import AddGaussianNoise

def evaluate_model():
    print("Starting Post-Training Evaluation on variable noise levels...")
    
    # Load Model
    model = DenoisingAutoencoder().to(config.DEVICE)
    model_path = os.path.join(config.MODELS_DIR, 'denoising_autoencoder.pth')
    if not os.path.exists(model_path):
        print(f"Error: Could not find trained model at {model_path}.")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval()
    
    # Prepare clean test dataset
    transform_clean = transforms.ToTensor()
    testset = datasets.MNIST(root=config.DATA_DIR, train=False, download=True, transform=transform_clean)
    
    # Grab a single batch of clean images for visualization
    test_loader = DataLoader(testset, batch_size=5, shuffle=True)
    clean_images, _ = next(iter(test_loader))
    
    # Noise levels requested: ranging up to 10%
    # We will test on 1%, 3%, 6%, and 10%
    noise_levels = [0.01, 0.03, 0.06, 0.10]
    
    fig, axes = plt.subplots(nrows=len(noise_levels) * 3, ncols=5, figsize=(15, 4 * len(noise_levels)))
    
    # Optional: adjust spacing
    plt.subplots_adjust(wspace=0.1, hspace=0.4)
    
    with torch.no_grad():
        for i, noise_factor in enumerate(noise_levels):
            # Create Noise Adder
            noise_adder = AddGaussianNoise(noise_factor=noise_factor)
            
            # Apply Noise
            noisy_images = noise_adder(clean_images)
            
            # Reconstruct
            outputs = model(noisy_images.to(config.DEVICE)).cpu()
            
            # Plot Results for this noise level
            row_idx = i * 3
            
            for j in range(5):
                # Original
                ax = axes[row_idx, j]
                ax.imshow(clean_images[j].squeeze(), cmap='gray')
                if j == 0:
                    ax.set_ylabel(f"Original", fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Noisy
                ax = axes[row_idx + 1, j]
                ax.imshow(noisy_images[j].squeeze(), cmap='gray')
                if j == 0:
                    ax.set_ylabel(f"Noise: {noise_factor*100:.1f}%", fontsize=12, color='red')
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Reconstructed
                ax = axes[row_idx + 2, j]
                ax.imshow(outputs[j].squeeze(), cmap='gray')
                if j == 0:
                    ax.set_ylabel(f"Reconstructed", fontsize=12, color='green')
                ax.set_xticks([])
                ax.set_yticks([])

    plt.suptitle("Evaluation Across Different Noise Levels", fontsize=16)
    save_path = os.path.join(config.ASSETS_DIR, 'evaluation_noise_levels.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    print(f"Evaluation complete! Results saved to: {save_path}")

if __name__ == '__main__':
    evaluate_model()
