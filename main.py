import argparse
import torch
from torch import nn
import numpy as np
import os
import json
from model import create_model
from data import load_dataset
from engine import train_one_epoch, validate
from torch import optim
from logger import setup_logger
from datetime import datetime
import random
from torchmetrics import Accuracy
from utils import save_checkpoint


def get_optimizer(model, trainloader, args):
    """Create optimizer and learning rate scheduler"""
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Create learning rate scheduler with warmup
    def warmup_cosine_schedule(step):
        if step < args.warmup_steps:
            return float(step) / float(max(1, args.warmup_steps))
        else:
            progress = float(step - args.warmup_steps) / float(
                max(1, args.epochs * len(trainloader) - args.warmup_steps)
            )
            return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_schedule)

    return optimizer, scheduler


def get_args_parser():
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d-%H-%M-%S")
    parser = argparse.ArgumentParser("ViT Training", add_help=False)

    # Model configuration
    parser.add_argument("--in_channels", default=3, type=int, help="Number of input image channels")
    parser.add_argument("--img_size", default=64, type=int, help="Input image size")
    parser.add_argument("--patch_size", default=8, type=int, help="Patch size")
    parser.add_argument("--embedding_dim", default=768, type=int, help="Dimension of token embeddings")
    parser.add_argument("--depth", default=12, type=int, help="Number of transformer layers")
    parser.add_argument("--num_heads", default=12, type=int, help="Number of attention heads")
    parser.add_argument("--mlp_ratio", default=4, type=int, help="Ratio of mlp hidden dim to embedding dim")
    parser.add_argument("--mlp_dim", default=3072, type=int, help="Ratio of mlp hidden dim to embedding dim")
    parser.add_argument("--num_classes", default=10, type=int, help="Number of classes for classification")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout rate")
    parser.add_argument("--attn_dropout_rate", default=0.0, type=float, help="Attention dropout rate")

    # Dataset parameters
    parser.add_argument("--dataset", default="eurosat", type=str, help="Dataset name")
    parser.add_argument("--data_dir", default="./data/EuroSAT_RGB", type=str, help="Path to dataset")

    # Training parameters
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs")
    parser.add_argument("--learning_rate", default=3e-4, type=float, help="Base learning rate")
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight decay")
    parser.add_argument("--warmup_steps", default=500, type=int, help="Number of warmup steps")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--save_dir", default=f"./checkpoints/{date_time_str}", type=str, help="Ckpt Save Location")

    # Saving parameters
    parser.add_argument("--save_freq", default=5, type=int, help="Frequency (in epochs) to save checkpoints")
    parser.add_argument("--print_freq", default=20, type=int, help="Frequency (in steps) to print training status")

    # Retrain
    parser.add_argument("--retrain", default=False, type=bool, help="Whether to resume training")
    parser.add_argument("--retrain_ckpt", default="", type=str, help="Which checkpoint to retrain?")
    # parser.add_argument("--retrain_epochs", default="", type=str, help="How many epochs for retraining?")

    return parser


def main(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # fix the seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logger(name="vit", log_dir=args.save_dir, timestamp=timestamp)
    logger.info(f"Directory '{args.save_dir}' created.")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load dataset
    trainloader, valloader = load_dataset(args)

    # Create model
    args.mlp_dim = args.mlp_ratio * args.embedding_dim
    model = create_model(args)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    model.to(device)

    # Define loss function, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    accuracy_metric = Accuracy(task="multiclass", num_classes=args.num_classes)
    optimizer, scheduler = get_optimizer(model, trainloader, args)
    best_val_acc = 0.0

    # Save configuration
    with open(os.path.join(args.save_dir, f"config_{timestamp}.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    start_epoch = 0

    if args.retrain:
        logger.info(f"Retraining ckpt {args.retrain_ckpt} for {args.epochs} epochs")
        ckpt = torch.load(args.retrain_ckpt)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_val_acc = ckpt["val_acc"]
        start_epoch = ckpt["epoch"]
        args.epochs = ckpt["epoch"] + args.epochs

    # Training loop
    logger.info("Starting training...")

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, criterion, accuracy_metric, trainloader, optimizer, scheduler, device, epoch + 1, logger, args
        )

        # Print epoch results
        logger.info(f"EPOCH {epoch + 1}, MEAN LOSS: {train_loss:.4f}, MEAN ACCURACY: {train_acc:.4f}")

        # Validate every `args.save_freq`
        if (epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs:
            val_loss, val_acc = validate(model, criterion, accuracy_metric, valloader, device)
            logger.info(f"VAL LOSS: {val_loss:.4f}, VAL ACC: {val_acc:.2f}%")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(model, optimizer, scheduler, epoch + 1, args, val_acc, name=f"VIT_bestvalacc_ckpt.pth")
                logger.info(f"[*] MODEL SAVED AT EPOCH {epoch + 1}: {val_acc:.2f}%")
            else:
                save_checkpoint(model, optimizer, scheduler, epoch + 1, args, val_acc, name=f"VIT_ckpt_{epoch + 1}.pth")
                logger.info(f"MODEL SAVED AT EPOCH {epoch + 1}: {val_acc:.2f}%")

    logger.info(f"BEST VALIDATION ACCURACY: {best_val_acc:.2f}%")
    logger.info("TRAINING COMPLETED!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ViT Training", parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
