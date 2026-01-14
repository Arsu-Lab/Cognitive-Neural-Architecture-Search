import torch
from torch.backends.mps import is_available
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import argparse
from tqdm import tqdm
import os
import sys
import wandb
import math

sys.path.append(".")
from evolution.architecture import Architecture


def accuracy_topk(output, target, topk=(1, 5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


from collections import OrderedDict


class Flatten(nn.Module):
    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):
    """
    Helper module that stores th"e current tensor. Useful for accessing by name
    """

    def forward(self, x):
        return x


class CORblock_S(nn.Module):
    scale = 4  # scale of the bottleneck convolution channels

    def __init__(self, in_channels, out_channels, times=1):
        super().__init__()

        self.times = times

        self.conv_input = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.skip = nn.Conv2d(
            out_channels, out_channels, kernel_size=1, stride=2, bias=False
        )
        self.norm_skip = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(
            out_channels, out_channels * self.scale, kernel_size=1, bias=False
        )
        self.nonlin1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels * self.scale,
            out_channels * self.scale,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.nonlin2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(
            out_channels * self.scale, out_channels, kernel_size=1, bias=False
        )
        self.nonlin3 = nn.ReLU(inplace=True)

        self.output = Identity()  # for an easy access to this block's output

        # need BatchNorm for each time step for training to work well
        for t in range(self.times):
            setattr(self, f"norm1_{t}", nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f"norm2_{t}", nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f"norm3_{t}", nn.BatchNorm2d(out_channels))

    def forward(self, inp):
        x = self.conv_input(inp)

        for t in range(self.times):
            if t == 0:
                skip = self.norm_skip(self.skip(x))
                self.conv2.stride = (2, 2)
            else:
                skip = x
                self.conv2.stride = (1, 1)

            x = self.conv1(x)
            x = getattr(self, f"norm1_{t}")(x)
            x = self.nonlin1(x)

            x = self.conv2(x)
            x = getattr(self, f"norm2_{t}")(x)
            x = self.nonlin2(x)

            x = self.conv3(x)
            x = getattr(self, f"norm3_{t}")(x)

            x += skip
            x = self.nonlin3(x)
            output = self.output(x)

        return output


def CORnet_S():
    model = nn.Sequential(
        OrderedDict(
            [
                (
                    "V1",
                    nn.Sequential(
                        OrderedDict(
                            [
                                (
                                    "conv1",
                                    nn.Conv2d(
                                        3,
                                        64,
                                        kernel_size=7,
                                        stride=2,
                                        padding=3,
                                        bias=False,
                                    ),
                                ),
                                ("norm1", nn.BatchNorm2d(64)),
                                ("nonlin1", nn.ReLU(inplace=True)),
                                (
                                    "pool",
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                ),
                                (
                                    "conv2",
                                    nn.Conv2d(
                                        64,
                                        64,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bias=False,
                                    ),
                                ),
                                ("norm2", nn.BatchNorm2d(64)),
                                ("nonlin2", nn.ReLU(inplace=True)),
                                ("output", Identity()),
                            ]
                        )
                    ),
                ),
                ("V2", CORblock_S(64, 128, times=2)),
                ("V4", CORblock_S(128, 256, times=4)),
                ("IT", CORblock_S(256, 512, times=2)),
                (
                    "decoder",
                    nn.Sequential(
                        OrderedDict(
                            [
                                ("avgpool", nn.AdaptiveAvgPool2d(1)),
                                ("flatten", Flatten()),
                                ("linear", nn.Linear(512, 1000)),
                                ("output", Identity()),
                            ]
                        )
                    ),
                ),
            ]
        )
    )

    # weight initialization
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
        # nn.Linear is missing here because I originally forgot
        # to add it during the training of this network
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch_config_file",
        type=str,
        required=True,
        help="Path to architecture JSON file",
    )
    parser.add_argument(
        "--epochs", type=int, default=120, help="Number of epochs to train"
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Warmup epochs")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/data/imagenet",
        help="Path to ImageNet dataset",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="Dropout rate for classifier"
    )
    parser.add_argument(
        "--grad_clip", type=float, default=1.0, help="Gradient clipping value"
    )
    parser.add_argument("--no_wandb", type=bool, default=False, help="Skip using wandb")

    args = parser.parse_args()
    arch_name = "EvoIT_modern"  # args.arch_config_file.split('/')[-1].split('.')[0]

    layers = [
        {
            "type": "conv",
            "out_channels": 384,
            "kernel_size": 7,
            "stride": 4,
            "padding": 0,
        },
        {"type": "pool", "kernel_size": 3, "stride": 2},
        {
            "type": "conv",
            "out_channels": 512,
            "kernel_size": 11,
            "stride": 1,
            "padding": 5,
        },
        {"type": "pool", "kernel_size": 3, "stride": 2},
        {
            "type": "conv",
            "out_channels": 512,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
        },
        {"type": "pool", "kernel_size": 3, "stride": 2},
        {
            "type": "conv",
            "out_channels": 512,
            "kernel_size": 9,
            "stride": 1,
            "padding": 4,
        },
        {
            "type": "conv",
            "out_channels": 512,
            "kernel_size": 5,
            "stride": 1,
            "padding": 2,
        },
        {
            "type": "conv",
            "out_channels": 512,
            "kernel_size": 5,
            "stride": 1,
            "padding": 2,
        },
    ]

    if not args.no_wandb:
        wandb.init(
            project="shallow-brain-evo-nas-nsd-imagenet", name=arch_name, config=args
        )

    # Set device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.mps.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Create architecture and build model
    architecture = Architecture(layers=layers)
    # model = CORnet_S().to(device) #vgg16(pretrained=False, num_classes=1000).to(device)
    model = architecture.build_model(device)
    model.apply(architecture.init_weights)

    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        dummy_output = model(dummy_input)
        if len(dummy_output.shape) == 4:  # Still conv features
            dummy_output = dummy_output.view(dummy_output.size(0), -1)
        final_features = dummy_output.shape[1]

    classifier = nn.Sequential(
        nn.Dropout(args.dropout),
        nn.Linear(final_features, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(args.dropout),
        nn.Linear(4096, 1000),
    ).to(device)
    model = nn.Sequential(model, nn.Flatten(), classifier).to(device)

    # Initialize weights
    model.apply(architecture.init_weights)

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Data transforms
    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Check if data path exists
    if not os.path.exists(args.data_path):
        print(
            f"Warning: Data path {args.data_path} does not exist. Skipping data loading."
        )
        print("Model architecture created successfully!")
        return

    # Load datasets
    train_path = os.path.join(args.data_path, "train")
    val_path = os.path.join(args.data_path, "val")

    if not os.path.exists(train_path):
        print(f"Warning: Training data path {train_path} does not exist.")
        print("Model architecture created successfully!")
        return

    train_dataset = ImageFolder(train_path, transform=train_transform)
    val_dataset = (
        ImageFolder(val_path, transform=val_transform)
        if os.path.exists(val_path)
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=32,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=32,
        )
        if val_dataset
        else None
    )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
    )

    # Learning rate scheduler with warmup
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, total_iters=args.warmup_epochs
    )
    main_scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[args.warmup_epochs],
    )

    print(f"Starting training for {args.epochs} epochs...")
    max_val_acc = 0.0
    max_val_acc5 = 0.0

    scaler = torch.amp.GradScaler("cuda")

    # Training loop
    for epoch in tqdm(range(args.epochs), position=0):
        # Training phase
        model.train()
        train_loss = 0.0
        train_acc1_total = 0.0
        train_acc5_total = 0.0
        train_samples = 0

        accum_steps = 2

        optimizer.zero_grad(set_to_none=True)

        train_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]", position=1
        )
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.float16):
                output = model(data)
                loss = criterion(output, target)

            scaler.scale(loss / accum_steps).backward()

            # update weights every `accum_steps` mini-batches
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(
                train_loader
            ):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # Calculate accuracies
            acc1, acc5 = accuracy_topk(output, target, topk=(1, 5))

            train_loss += loss.item()
            train_acc1_total += acc1.item() * target.size(0)
            train_acc5_total += acc5.item() * target.size(0)
            train_samples += target.size(0)

            # Update progress bar
            current_acc1 = train_acc1_total / train_samples
            current_acc5 = train_acc5_total / train_samples
            avg_loss = train_loss / (batch_idx + 1)
            train_pbar.set_postfix(
                {
                    "Loss": f"{avg_loss:.4f}",
                    "Acc@1": f"{current_acc1:.2f}%",
                    "Acc@5": f"{current_acc5:.2f}%",
                }
            )

            # Log every 100 batches instead of every batch
            if batch_idx % 100 == 0:
                wandb.log(
                    {
                        "train_loss": loss.item(),
                        "train_acc1": current_acc1,
                        "train_acc5": current_acc5,
                        "lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                    }
                )

        scheduler.step()

        # Validation phase
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_acc1_total = 0.0
            val_acc5_total = 0.0
            val_samples = 0

            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Val]")
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(val_pbar):
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)

                    # Calculate accuracies
                    acc1, acc5 = accuracy_topk(output, target, topk=(1, 5))

                    val_loss += loss.item()
                    val_acc1_total += acc1.item() * target.size(0)
                    val_acc5_total += acc5.item() * target.size(0)
                    val_samples += target.size(0)

                    # Update progress bar
                    current_val_acc1 = val_acc1_total / val_samples
                    current_val_acc5 = val_acc5_total / val_samples
                    avg_val_loss = val_loss / (batch_idx + 1)
                    val_pbar.set_postfix(
                        {
                            "Loss": f"{avg_val_loss:.4f}",
                            "Acc@1": f"{current_val_acc1:.2f}%",
                            "Acc@5": f"{current_val_acc5:.2f}%",
                        }
                    )

        # Print epoch summary
        train_acc1_final = train_acc1_total / train_samples
        train_acc5_final = train_acc5_total / train_samples
        tqdm.write(f"\nEpoch {epoch + 1}/{args.epochs} Summary:")
        tqdm.write(
            f"Train Loss: {train_loss / len(train_loader):.4f}, Train Acc@1: {train_acc1_final:.2f}%, Train Acc@5: {train_acc5_final:.2f}%"
        )

        # Log epoch metrics
        epoch_metrics = {
            "epoch": epoch,
            "train_loss_epoch": train_loss / len(train_loader),
            "train_acc1_epoch": train_acc1_final,
            "train_acc5_epoch": train_acc5_final,
            "lr_epoch": optimizer.param_groups[0]["lr"],
        }

        if val_loader:
            val_acc1_final = val_acc1_total / val_samples
            val_acc5_final = val_acc5_total / val_samples
            tqdm.write(
                f"Val Loss: {val_loss / len(val_loader):.4f}, Val Acc@1: {val_acc1_final:.2f}%, Val Acc@5: {val_acc5_final:.2f}%"
            )

            epoch_metrics.update(
                {
                    "val_loss_epoch": val_loss / len(val_loader),
                    "val_acc1_epoch": val_acc1_final,
                    "val_acc5_epoch": val_acc5_final,
                }
            )

            # Save best models
            if val_acc1_final > max_val_acc:
                tqdm.write(
                    f"New best val top-1 accuracy: {val_acc1_final:.2f}% (prev: {max_val_acc:.2f}%)"
                )
                max_val_acc = val_acc1_final
                torch.save(model.state_dict(), f"best_acc1_{arch_name}.pth")

            if val_acc5_final > max_val_acc5:
                tqdm.write(
                    f"New best val top-5 accuracy: {val_acc5_final:.2f}% (prev: {max_val_acc5:.2f}%)"
                )
                max_val_acc5 = val_acc5_final
                torch.save(model.state_dict(), f"best_acc5_{arch_name}.pth")

        if not args.no_wandb:
            wandb.log(epoch_metrics)


if __name__ == "__main__":
    main()
