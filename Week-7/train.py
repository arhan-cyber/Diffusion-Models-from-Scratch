import os
import csv
import yaml
import argparse
import time
from pathlib import Path
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset import EMNISTLettersDataset
from model import UNet
from diffusion import Diffusion

def get_args():
    parser = argparse.ArgumentParser(description="Train Class-Conditional DDPM on EMNIST Letters")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb-key", type=str, default=None, help="Weights & Biases API Key for programmatic login")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run name for logging")
    return parser.parse_args()

def save_samples(model, diffusion, epoch, results_dir, device):
    """
    Generates a full A-Z grid of letters (26 samples) with a CFG scale of 3.0 to inspect quality.
    """
    labels = torch.arange(26, device=device)
    samples = diffusion.sample(
        model=model,
        n=26,
        image_size=28,
        channels=1,
        labels=labels,
        cfg_scale=3.0
    )
    output_path = results_dir / f"samples_epoch_{epoch}.png"
    save_image(samples, output_path, nrow=7)
    print(f"Saved epoch {epoch} generation samples to {output_path}")
    return samples

def save_loss_curve(loss_history, output_path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("EMNIST Letters Diffusion Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    args = get_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    epochs = args.epochs if args.epochs is not None else config["training"]["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else config["training"]["batch_size"]
    lr = args.lr if args.lr is not None else config["training"]["learning_rate"]
    noise_steps = config["training"]["noise_steps"]
    sample_every = config["training"]["sample_every"]
    checkpoint_every = config["training"]["checkpoint_every"]
    
    # Setup directories
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Initialize logger
    use_wandb = args.wandb or config["training"].get("use_wandb", False) or (args.wandb_key is not None)
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            if args.wandb_key:
                wandb.login(key=args.wandb_key)
            # Check authentication
            if wandb.api.api_key is None:
                print("Weights & Biases API Key not found. Falling back to local logging.")
                use_wandb = False
            else:
                run_name = args.run_name or f"emnist-letters-cfg-{int(time.time())}"
                wandb_run = wandb.init(
                    project=config["training"].get("wandb_project", "diffusion-letters"),
                    name=run_name,
                    config={
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "learning_rate": lr,
                        "noise_steps": noise_steps,
                        "base_channels": config["model"]["base_channels"],
                        "depth": config["model"]["depth"],
                        "emb_dim": config["model"]["emb_dim"]
                    }
                )
        except ImportError:
            print("wandb package not installed. Falling back to local logging.")
            use_wandb = False
            
    # Device setup
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load custom EMNIST letters dataset
    dataset = EMNISTLettersDataset(train=True, augment=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    
    # Initialize UNet and Diffusion Process (26 classes representing A-Z letters)
    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=config["model"]["base_channels"],
        depth=config["model"]["depth"],
        emb_dim=config["model"]["emb_dim"],
        num_classes=26
    ).to(device)
    
    diffusion = Diffusion(
        noise_steps=noise_steps,
        device=device
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    
    # Local log setup
    log_file = results_dir / "experiment_log.csv"
    with open(log_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Average Loss", "Time (seconds)"])
        
    loss_history = []
    total_start_time = time.time()
    
    print(f"Starting training for {epochs} epochs on EMNIST letters...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Classifier-Free Guidance: Randomly drop label conditioning (15% rate)
            # Null label is set to index 26 (num_classes)
            drop_mask = torch.rand(labels.shape[0], device=device) < 0.15
            labels[drop_mask] = 26
            
            t = diffusion.sample_timesteps(images.shape[0])
            x_t, noise = diffusion.noise_images(images, t)
            
            predicted_noise = model(x_t, t, labels)
            loss = mse_loss(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(batch_loss=loss.item())
            
        avg_loss = epoch_loss / len(dataloader)
        epoch_duration = time.time() - epoch_start
        loss_history.append(avg_loss)
        
        print(f"Epoch {epoch + 1}/{epochs} Completed | Avg Loss: {avg_loss:.6f} | Time: {epoch_duration:.2f}s")
        
        # Log metrics
        with open(log_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_loss, epoch_duration])
            
        if use_wandb:
            wandb.log({
                "loss": avg_loss,
                "epoch_time": epoch_duration,
                "epoch": epoch + 1
            })
            
        # Periodically sample visual outputs
        if (epoch + 1) % sample_every == 0 or (epoch + 1) == epochs:
            samples = save_samples(model, diffusion, epoch + 1, results_dir, device)
            if use_wandb:
                # Log samples to W&B
                images_logged = wandb.Image(samples, caption=f"Epoch {epoch + 1} Generations")
                wandb.log({"samples": images_logged})
                
        # Periodically save checkpoints
        if (epoch + 1) % checkpoint_every == 0 or (epoch + 1) == epochs:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "config": model.get_config()
            }, checkpoint_path)
            print(f"Saved model checkpoint to {checkpoint_path}")
            
    # Save final model
    final_model_path = checkpoint_dir / "final_model.pt"
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "config": model.get_config()
    }, final_model_path)
    print(f"Saved final model weights to {final_model_path}")
    
    # Save final loss curve plot
    save_loss_curve(loss_history, results_dir / "training_curves.png")
    
    total_duration = time.time() - total_start_time
    print(f"Training completed successfully in {total_duration/60:.2f} minutes!")
    
    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
