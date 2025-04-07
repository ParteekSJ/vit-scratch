import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from .eurosat_dataset import EuroSATDataset
import os


def load_dataset(args):
    """Load and prepare dataset based on args"""
    print(f"Loading dataset: {args.dataset}")

    # Define transforms
    if args.dataset.lower() == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    else:  # ImageNet defaults
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(args.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    transform_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    # Load appropriate dataset
    if args.dataset.lower() == "cifar10":
        trainset = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=True, download=False, transform=transform_train
        )
        valset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=False, transform=transform_val)
        args.num_classes = 10

    elif args.dataset.lower() == "cifar100":
        trainset = torchvision.datasets.CIFAR100(
            root=args.data_dir, train=True, download=False, transform=transform_train
        )
        valset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=False, transform=transform_val)
        args.num_classes = 100

    elif args.dataset.lower() == "imagenet":
        trainset = torchvision.datasets.ImageFolder(os.path.join(args.data_dir, "train"), transform=transform_train)
        valset = torchvision.datasets.ImageFolder(os.path.join(args.data_dir, "val"), transform=transform_val)
        args.num_classes = 1000

    elif args.dataset.lower() == "eurosat":
        trainset = EuroSATDataset(root_dir=args.data_dir, split="train", transforms=transform_train)
        valset = EuroSATDataset(root_dir=args.data_dir, split="val", transforms=transform_val)
        args.num_classes = 10

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Create data loaders
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    return trainloader, valloader
