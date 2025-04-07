from model import ViT
import torch
from typing import Iterable
from torchmetrics.classification.accuracy import MulticlassAccuracy


def train_one_epoch(
    model: ViT,
    criterion: torch.nn.Module,
    accuracy: MulticlassAccuracy,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    logger,
    args,
):
    train_losses = []
    train_accs = []

    model.train()
    for idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)  # [B, 3, 224, 224], [B,]

        outputs = model(inputs)  # [B, 10] -- distribution over the 10 classes
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        predictions = outputs.argmax(dim=-1)

        acc = accuracy(predictions.view(-1), targets.view(-1))  # Flatten for token-level accuracy

        train_losses.append(loss.item())
        train_accs.append(acc.item())

        if idx % args.print_freq == 0:
            logger.info(f"EPOCH {epoch}, STEP [{idx}/{len(data_loader)}], LOSS: {loss:.4f}, ACCURACY: {acc:.4f}")

    return torch.mean(torch.Tensor(train_losses)).item(), torch.mean(torch.Tensor(train_accs)).item()


@torch.no_grad
def validate(
    model: ViT,
    criterion: torch.nn.Module,
    accuracy: MulticlassAccuracy,
    data_loader: Iterable,
    device: torch.device,
):
    val_losses, val_accs = [], []
    model.eval()
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)  # [B, 3, 224, 224], [B,]

        outputs = model(inputs)  # [B, 10] -- distribution over the 10 classes
        predictions = outputs.argmax(dim=-1)

        loss = criterion(outputs, targets)
        acc = accuracy(predictions.view(-1), targets.view(-1))

        val_losses.append(loss.item())
        val_accs.append(acc.item())

    return torch.mean(torch.Tensor(val_losses)).item(), torch.mean(torch.Tensor(val_accs)).item()
