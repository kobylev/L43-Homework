def train_model():
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * noisy_imgs.size(0)
            
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for noisy_imgs, clean_imgs in val_loader:
                noisy_imgs, clean_imgs = noisy_imgs.to(config.DEVICE), clean_imgs.to(config.DEVICE)
                
                outputs = model(noisy_imgs)
                loss = criterion(outputs, clean_imgs)
                running_val_loss += loss.item() * noisy_imgs.size(0)
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch [{epoch+1}/{config.EPOCHS}], Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")
        
    print("Training Complete. Saving Model...")
    torch.save(model.state_dict(), os.path.join(config.MODELS_DIR, 'denoising_autoencoder.pth'))
    
    # Plotting Training Curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config.ASSETS_DIR, 'training_curve.png'))
    plt.close()
    
    return model

if __name__ == '__main__':
    train_model()
