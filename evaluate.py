def evaluate_model():
                clean_images = clean_images.to(config.DEVICE)
                noisy_images = noise_adder(clean_images)
                
                outputs = model(noisy_images)
                
                # Move to CPU for scikit-image metrics
                clean_np = clean_images.cpu().numpy()
                out_np = outputs.cpu().numpy()
                
                # Calculate batch metrics
                for c, o in zip(clean_np, out_np):
                    c_squeeze = c.squeeze()
                    o_squeeze = o.squeeze()
                    
                    mse_val = np.mean((c_squeeze - o_squeeze) ** 2)
                    p_val = psnr(c_squeeze, o_squeeze, data_range=1.0)
                    s_val = ssim(c_squeeze, o_squeeze, data_range=1.0)
                    
                    mse_list.append(mse_val)
                    psnr_list.append(p_val)
                    ssim_list.append(s_val)
                    
            results.append({
                'noise_level': noise_factor * 100,
                'mse_mean': np.mean(mse_list),
                'mse_std': np.std(mse_list),
                'psnr_mean': np.mean(psnr_list),
                'psnr_std': np.std(psnr_list),
                'ssim_mean': np.mean(ssim_list),
                'ssim_std': np.std(ssim_list)
            })

    # Saving CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(config.ASSETS_DIR, 'evaluation_stats.csv')
    df.to_csv(csv_path, index=False)
    print(f"Metrics saved to {csv_path}")
    
    # Plotting Metrics vs Noise
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    noise_percentages = df['noise_level']
    
    # MSE Plot
    axes[0].plot(noise_percentages, df['mse_mean'], color='red')
    axes[0].fill_between(noise_percentages, df['mse_mean'] - df['mse_std'], df['mse_mean'] + df['mse_std'], color='red', alpha=0.2)
    axes[0].set_title('MSE vs Noise %')
    axes[0].set_xlabel('Noise level %')
    axes[0].set_ylabel('MSE (lower is better)')
    
    # PSNR Plot
    axes[1].plot(noise_percentages, df['psnr_mean'], color='blue')
    axes[1].fill_between(noise_percentages, df['psnr_mean'] - df['psnr_std'], df['psnr_mean'] + df['psnr_std'], color='blue', alpha=0.2)
    axes[1].set_title('PSNR vs Noise %')
    axes[1].set_xlabel('Noise level %')
    axes[1].set_ylabel('PSNR (dB) (higher is better)')
    
    # SSIM Plot
    axes[2].plot(noise_percentages, df['ssim_mean'], color='green')
    axes[2].fill_between(noise_percentages, df['ssim_mean'] - df['ssim_std'], df['ssim_mean'] + df['ssim_std'], color='green', alpha=0.2)
    axes[2].set_title('SSIM vs Noise %')
    axes[2].set_xlabel('Noise level %')
    axes[2].set_ylabel('SSIM (higher is better)')
    
    plt.tight_layout()
    metrics_path = os.path.join(config.ASSETS_DIR, 'metrics_vs_noise.png')
    plt.savefig(metrics_path)
    plt.close()
    print(f"Metrics plot saved to: {metrics_path}")

    # Generate the Sample Reconstructions Grid (similar to before)
    print("Generating Sample Reconstructions Grid...")
    sample_loader = DataLoader(testset, batch_size=4, shuffle=True)
    clean_images, _ = next(iter(sample_loader))
    
    sample_noise_levels = [0.0001, 0.05, 0.10]
    fig, axes = plt.subplots(nrows=len(sample_noise_levels) * 3, ncols=4, figsize=(10, 3 * len(sample_noise_levels)))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    
    with torch.no_grad():
        for i, nf in enumerate(sample_noise_levels):
            n_adder = AddGaussianNoise(noise_factor=nf)
            n_imgs = n_adder(clean_images)
            outs = model(n_imgs.to(config.DEVICE)).cpu()
            
            row_idx = i * 3
            for j in range(4):
                # Clean
                ax = axes[row_idx, j]
                ax.imshow(clean_images[j].squeeze(), cmap='gray')
                if j == 0: ax.set_ylabel('Clean', fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])
                
                # Noisy
                ax = axes[row_idx + 1, j]
                ax.imshow(n_imgs[j].squeeze(), cmap='gray')
                if j == 0: ax.set_ylabel(f'Noisy ({nf*100:.2f}%)', fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])
                
                # Recon
                ax = axes[row_idx + 2, j]
                ax.imshow(outs[j].squeeze(), cmap='gray')
                if j == 0: ax.set_ylabel('Reconstructed', fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle("Sample Reconstructions", fontsize=16)
    grid_path = os.path.join(config.ASSETS_DIR, 'sample_reconstructions.png')
    plt.savefig(grid_path, bbox_inches='tight')
    plt.close()
    
    print(f"Sample grid saved to: {grid_path}")

if __name__ == '__main__':
    evaluate_model()
