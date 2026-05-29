import torch

class NoisyDataset(torch.utils.data.Dataset):

    def __init__(self,
                 base_dataset,
                 noise_min=0.05,
                 noise_max=0.5):

        self.dataset = base_dataset
        self.noise_min = noise_min
        self.noise_max = noise_max

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        img, _ = self.dataset[idx]

        sigma = torch.empty(1).uniform_(
            self.noise_min,
            self.noise_max
        )

        noise = torch.randn_like(img) * sigma

        noisy = img + noise
        noisy = torch.clamp(noisy, 0, 1)

        return noisy, img