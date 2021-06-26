import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from early_stopping import EarlyStopping
from model import WaveformRecognition

OLD_LABELS = ['backward', 'down', 'eight', 'five', 'forward', 'four', 'go',
              'left', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'six',
              'stop', 'three', 'two', 'up', 'yes', 'zero']

NEW_LABELS = ['train', 'velocity', 'direction', 'start']

LABELS = OLD_LABELS + NEW_LABELS


class ModelManager:
    def __init__(self, epochs=10, batch_size=256, train_set=None, test_set=None,
                 num_workers=0, pin_memory=False):
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if train_set is not None:
            self.train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=self.__collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        if test_set is not None:
            self.test_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=self.__collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

        self.model = WaveformRecognition(n_output=len(LABELS)).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.01, weight_decay=0.0001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)
        self.early_stopping = EarlyStopping(patience=10, verbose=True)

    def load_model(self, model_name):
        self.model.load_state_dict(torch.load(model_name))

    def train(self):
        for epoch in range(self.epochs):
            train_loss = self.__train_step(epoch)
            test_loss = self.__test_step(epoch)
            self.__print_epoch_info(epoch, train_loss, test_loss)

            self.early_stopping(test_loss, self.model)
            if self.early_stopping.early_stop:
                print(f"Early stopping at {epoch}")
                break

            self.scheduler.step()

        name = f'trained_models\\model_all_16k_epochs_{self.epochs}.pt'
        torch.save(self.model.state_dict(), name)

    def predict(self, waveform):
        self.model.eval()
        waveform = waveform.to(self.device)
        waveform = waveform.unsqueeze(0)
        return self.model(waveform)

    def __train_step(self, epoch):
        losses = []
        self.model.train()
        for batch_idx, (waveform, label) in enumerate(self.train_loader):
            waveform = waveform.to(self.device)
            label = label.to(self.device)

            output = self.model(waveform)
            loss = F.cross_entropy(output.squeeze(), label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

            if batch_idx % 20 == 0:
                print(f"Train Epoch: {epoch} [{batch_idx * len(waveform)}/{len(self.train_loader.dataset)} "
                      f"({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        return np.average(losses)

    def __test_step(self, epoch):
        losses = []
        self.model.eval()
        correct = 0
        for waveform, label in self.test_loader:
            waveform = waveform.to(self.device)
            label = label.to(self.device)

            output = self.model(waveform)
            loss = F.cross_entropy(output.squeeze(), label)

            predictions = output.argmax(dim=-1)
            correct += self.__number_of_correct(predictions, label)
            losses.append(loss.item())

        print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(self.test_loader.dataset)} "
              f"({100. * correct / len(self.test_loader.dataset):.0f}%)\n")

        return np.average(losses)

    @staticmethod
    def label_to_index(word):
        return torch.tensor(LABELS.index(word))

    @staticmethod
    def index_to_label(index):
        return LABELS[index]

    @staticmethod
    def __print_epoch_info(epoch, train_loss, test_loss):
        print(f'Epoch {epoch} finished. ' +
              f'Train loss: {train_loss:.5f} ' +
              f'Test loss: {test_loss:.5f}\n')

    def __collate_fn(self, batch):
        tensors, targets = [], []
        for waveform, _, label, *_ in batch:
            tensors += [waveform]
            targets += [self.label_to_index(label)]

        tensors = self.__pad_sequence(tensors)
        targets = torch.stack(targets)

        return tensors, targets

    @staticmethod
    def __pad_sequence(batch):
        batch = [item.t() for item in batch]
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
        return batch.permute(0, 2, 1)

    @staticmethod
    def __number_of_correct(predictions, target):
        return predictions.squeeze().eq(target).sum().item()
