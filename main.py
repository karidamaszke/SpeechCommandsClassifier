import torch

from dataset import Dataset
from model_manager import ModelManager


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = Dataset("training")
    test_set = Dataset("testing")

    if device == "cuda":
        num_workers = 1
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    model_manager = ModelManager(
        epochs=120,
        batch_size=256,
        train_set=train_set,
        test_set=test_set,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    model_manager.train()


if __name__ == '__main__':
    main()
