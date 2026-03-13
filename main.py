from train import train_model
from evaluate import evaluate_model

def main():
    print("=== Denoising Autoencoder Pipeline ===")
    
    # 1. Train the model
    print("\n--- Phase 1: Training ---")
    train_model()
    
    # 2. Evaluate the model at various noise levels
    print("\n--- Phase 2: Evaluation ---")
    evaluate_model()
    
    print("\n=== Pipeline Execution Finished Successfully ===")

if __name__ == '__main__':
    main()
