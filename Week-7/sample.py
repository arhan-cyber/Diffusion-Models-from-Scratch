import argparse
from pathlib import Path
import torch
from torchvision.utils import save_image

from model import UNet
from diffusion import Diffusion

def get_args():
    parser = argparse.ArgumentParser(description="Generate samples from trained EMNIST Diffusion model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/final_model.pt", help="Path to checkpoint .pt file")
    parser.add_argument("--letters", type=str, default="all", help="Comma-separated letters (e.g. A,B,C,D) or 'all' for full A-Z grid")
    parser.add_argument("--cfg", type=float, default=3.0, help="Classifier-free guidance scale (w >= 0)")
    parser.add_argument("--num-samples-per-class", type=int, default=8, help="Number of samples to generate per requested letter")
    parser.add_argument("--output", type=str, default="results/generated_letters.png", help="Path to save output image grid")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for generation")
    return parser.parse_args()

def main():
    args = get_args()
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint info
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    
    # Instantiate and load model weights
    model = UNet(
        in_channels=config.get("in_channels", 1),
        out_channels=config.get("out_channels", 1),
        base_channels=config.get("base_channels", 64),
        depth=config.get("depth", 3),
        emb_dim=config.get("emb_dim", 256),
        num_classes=config.get("num_classes", 26)
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Model weights successfully loaded.")
    
    diffusion = Diffusion(
        noise_steps=1000,
        device=device
    )
    
    # Parse target letters
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    letter_to_class = {char: idx for idx, char in enumerate(alphabet)}
    
    if args.letters.lower() == "all":
        # Generate a neat grid containing one of each letter A-Z
        target_letters = list(alphabet)
        n_samples_per_class = 1
        num_cols = 7
    else:
        target_letters = [c.strip().upper() for c in args.letters.split(",") if c.strip().upper() in letter_to_class]
        n_samples_per_class = args.num_samples_per_class
        num_cols = n_samples_per_class
        
    if not target_letters:
        print("Error: No valid letters specified. Use letters from A to Z.")
        return
        
    print(f"Generating samples for letters: {', '.join(target_letters)} (CFG scale: {args.cfg})")
    
    # Build the label tensor
    labels_list = []
    for letter in target_letters:
        class_id = letter_to_class[letter]
        labels_list.extend([class_id] * n_samples_per_class)
        
    labels = torch.tensor(labels_list, dtype=torch.long, device=device)
    num_samples = len(labels)
    
    # Generate samples
    samples = diffusion.sample(
        model=model,
        n=num_samples,
        image_size=28,
        channels=1,
        labels=labels,
        cfg_scale=args.cfg
    )
    
    # Save the output grid
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    save_image(samples, output_path, nrow=num_cols)
    print(f"Successfully saved generated image grid to: {output_path}")

if __name__ == "__main__":
    main()
