import os
from torchvision import datasets

def prepare_data(data_dir="./data"):
    print(f"Downloading/verifying EMNIST letters dataset in '{data_dir}'...")
    
    # EMNIST letters split download
    dataset = datasets.EMNIST(
        root=data_dir,
        split="letters",
        train=True,
        download=True
    )
    
    print(f"Successfully verified training set: {len(dataset)} samples.")
    
    # Verify test set
    test_dataset = datasets.EMNIST(
        root=data_dir,
        split="letters",
        train=False,
        download=True
    )
    print(f"Successfully verified test set: {len(test_dataset)} samples.")
    print("EMNIST classes found:", len(dataset.classes))
    print("Data preparation complete!")

if __name__ == "__main__":
    # Ensure working directory is correct relative to run context
    prepare_data()
