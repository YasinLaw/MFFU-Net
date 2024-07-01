import torch

if __name__ == "__main__":
    real_label = torch.full(
        [8, 1],
        1.0,
        dtype=torch.float32,
    )
    print(real_label)
