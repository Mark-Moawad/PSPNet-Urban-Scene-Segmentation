"""
PSPNet Road Segmentation for Cityscapes Dataset
================================================
This script implements:
- PSPNet model with ResNet50 encoder
- Training with auxiliary branch (matching original paper)
- Evaluation on validation set
- Video inference for autonomous driving scenarios
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from torchvision import models

from utils import (
    CityscapesDataset,
    MeanIoU,
    train_epoch,
    validate_epoch,
    plot_training_curves,
    save_predictions_grid,
    process_video
)

# ==================== Configuration ====================
# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
if not torch.cuda.is_available():
    print("WARNING: CUDA not available. Training will be VERY slow on CPU!")

# Dataset parameters (matching original PSPNet paper)
NUM_CLASSES = 19  # Cityscapes 19 classes
# INPUT_HEIGHT = 713  # Original paper (slow but best accuracy)
# INPUT_WIDTH = 713
INPUT_HEIGHT = 512  # Balanced resolution - much faster than 713 in the original paper
INPUT_WIDTH = 512
# BATCH_SIZE = 8  # Good balance for less powerful GPUs at 512x512 resolution with random crop
BATCH_SIZE = 16 # Increased batch size for 512x512 resolution

# Training parameters (following PSPNet paper)
LEARNING_RATE = 0.01  # Base LR with polynomial decay
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001
POWER = 0.9  # For polynomial LR scheduler
MAX_EPOCHS = 200  # Paper uses ~200 epochs for full convergence
AUX_WEIGHT = 0.4  # Auxiliary loss weight

# Paths
DATA_ROOT = Path("data/dataset")
MODELS_DIR = Path("data/models")
OUTPUTS_DIR = Path("data/outputs")
PROCESSED_DIR = Path("data/processed")
VIDEOS_DIR = Path("data/testing_videos")

# Create directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ==================== Model Definition ====================
class PyramidPoolingModule(nn.Module):
    """
    Pyramid Pooling Module (PPM) from the PSPNet paper.
    
    Pools features at 4 different scales (1x1, 2x2, 3x3, 6x6) and
    concatenates them with the original feature map for multi-scale context.
    """
    def __init__(self, in_channels, out_channels=512, pool_sizes=(1, 2, 3, 6)):
        super().__init__()
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Create a conv layer for each pooling level
        self.pool_convs = nn.ModuleList()
        for pool_size in pool_sizes:
            self.pool_convs.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_size),
                    nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(self.out_channels),
                    nn.ReLU(inplace=True)
                )
            )

    def forward(self, x):
        # Original feature map
        original_features = [x]
        
        # Get height and width of input feature map
        h, w = x.shape[2:]

        # Run each pooling level
        for pool_conv in self.pool_convs:
            pooled_features = pool_conv(x)
            # Upsample pooled features to original size
            upsampled_features = F.interpolate(
                pooled_features, 
                size=(h, w), 
                mode='bilinear', 
                align_corners=False
            )
            original_features.append(upsampled_features)
        
        # Concatenate all features (original + 4 pooled levels)
        # Output: in_channels + 4 * out_channels
        return torch.cat(original_features, dim=1)


class PSPNet(nn.Module):
    """
    PSPNet (Pyramid Scene Parsing Network) following the original paper.
    
    Architecture:
    - ResNet50 backbone with dilated convolutions in layer3 and layer4
    - Pyramid Pooling Module after layer4 (2048 channels)
    - Main classifier head with PPM features
    - Auxiliary classifier head from layer3 (1024 channels) for training
    
    Reference: Zhao et al., "Pyramid Scene Parsing Network", CVPR 2017
    """
    def __init__(self, num_classes=19, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        
        # Load ResNet50 backbone with dilated convolutions
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet50(
            weights=weights,
            replace_stride_with_dilation=[False, True, True]  # Dilate layer3 and layer4
        )
        
        # Extract ResNet layers (exclude avgpool and fc)
        self.backbone_layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.backbone_layer1 = resnet.layer1  # 256 channels
        self.backbone_layer2 = resnet.layer2  # 512 channels
        self.backbone_layer3 = resnet.layer3  # 1024 channels - for auxiliary
        self.backbone_layer4 = resnet.layer4  # 2048 channels - for main
        
        # Pyramid Pooling Module
        ppm_in_channels = 2048
        self.ppm = PyramidPoolingModule(in_channels=ppm_in_channels, out_channels=512)
        
        # PPM output: 2048 + 4 * 512 = 4096 channels
        ppm_out_channels = ppm_in_channels + 4 * 512

        # Main classifier head
        self.main_head = nn.Sequential(
            nn.Conv2d(ppm_out_channels, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )
        
        # Auxiliary classifier head (from layer3)
        self.aux_head = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # Store original input size for final upsampling
        input_size = x.shape[2:]
        
        # Backbone forward pass
        x = self.backbone_layer0(x)
        x = self.backbone_layer1(x)
        x = self.backbone_layer2(x)
        x_aux = self.backbone_layer3(x)   # Features for auxiliary loss (1024 channels)
        x_main = self.backbone_layer4(x_aux)  # Features for main loss (2048 channels)
        
        # Main branch: PPM + classifier
        x = self.ppm(x_main)
        x = self.main_head(x)
        # Upsample to original input size
        main_output = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)

        # Auxiliary branch (only during training)
        if self.training:
            aux = self.aux_head(x_aux)
            # Upsample to original input size
            aux_output = F.interpolate(aux, size=input_size, mode='bilinear', align_corners=False)
            
            # Return both outputs as tuple
            return main_output, aux_output
        else:
            # During evaluation, return only main output
            return main_output


def create_model(num_classes=NUM_CLASSES):
    """Create PSPNet model with auxiliary branch."""
    model = PSPNet(
        num_classes=num_classes,
        pretrained=True
    )
    return model.to(DEVICE)


# ==================== Training ====================
def train_model(model, train_loader, val_loader, num_epochs=MAX_EPOCHS):
    """
    Train PSPNet with auxiliary branch.
    
    Args:
        model: PSPNet model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        
    Returns:
        model: Trained model
        history: Training history dict
    """
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # Ignore unlabeled pixels
    
    # Optimizer: SGD with momentum (as per PSPNet paper)
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler: Polynomial decay
    # LR = base_lr * (1 - iter/max_iter)^power
    total_iters = len(train_loader) * num_epochs
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda iter: (1 - iter / total_iters) ** POWER
    )
    
    # Metrics
    train_metric = MeanIoU(num_classes=NUM_CLASSES)
    val_metric = MeanIoU(num_classes=NUM_CLASSES)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_iou': [],
        'val_iou': [],
        'lr': []
    }
    
    best_val_iou = 0.0
    
    print("\n" + "="*50)
    print("Starting PSPNet Training")
    print("="*50)
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Device: {DEVICE}")
    print("="*50 + "\n")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Training phase
        train_loss, train_iou = train_epoch(
            model, train_loader, criterion, optimizer, train_metric,
            NUM_CLASSES, DEVICE, scheduler, aux_weight=AUX_WEIGHT
        )
        
        # Validation phase
        val_loss, val_iou = validate_epoch(
            model, val_loader, criterion, val_metric, NUM_CLASSES, DEVICE
        )
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val IoU:   {val_iou:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            model_path = MODELS_DIR / "PSPNet_baseline.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'val_loss': val_loss,
            }, model_path)
            print(f"Best model saved! (Val IoU: {val_iou:.4f})")
    
    print("\n" + "="*50)
    print(f"Training Complete! Best Val IoU: {best_val_iou:.4f}")
    print("="*50 + "\n")
    
    return model, history


# ==================== Main ====================
def main():
    """Main execution function."""
    
    # Configuration flags
    train_model_flag = False  # Set to True to train
    evaluate_model_flag = False  # Set to True to evaluate on validation set
    visualize_flag = False  # Set to True to generate prediction visualizations
    process_videos_flag = True  # Set to True to process videos
    
    print("\n" + "="*60)
    print("PSPNet Road Segmentation Pipeline")
    print("Dataset: Cityscapes (19 classes)")
    print("="*60 + "\n")
    
    # Load datasets
    print("Loading Cityscapes dataset...")
    train_dataset = CityscapesDataset(
        root_dir=DATA_ROOT,
        split='train',
        crop_size=(INPUT_HEIGHT, INPUT_WIDTH),
        use_random_crop=True  # Random crop during training
    )
    
    val_dataset = CityscapesDataset(
        root_dir=DATA_ROOT,
        split='val',
        crop_size=(INPUT_HEIGHT, INPUT_WIDTH),
        use_random_crop=False  # Center crop during validation
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}\n")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Windows compatibility
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Create model
    print("Creating PSPNet model...")
    model = create_model()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")
    
    # Training
    if train_model_flag:
        print("="*60)
        print("Training PSPNet...")
        print("="*60)
        model, history = train_model(model, train_loader, val_loader, MAX_EPOCHS)
        
        # Plot training curves
        print("\nGenerating training curves...")
        plot_training_curves(history, save_path=OUTPUTS_DIR / "PSPNet_baseline_training_curves.png")
        print(f"Training curves saved to {OUTPUTS_DIR / 'PSPNet_baseline_training_curves.png'}")
    
    # Evaluation
    if evaluate_model_flag:
        print("\n" + "="*60)
        print("Evaluating on Validation Set...")
        print("="*60)
        
        # Load best model
        checkpoint = torch.load(MODELS_DIR / "PSPNet_baseline.pt", weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        criterion = nn.CrossEntropyLoss(ignore_index=255)
        val_metric = MeanIoU(num_classes=NUM_CLASSES)
        
        val_loss, val_iou = validate_epoch(
            model, val_loader, criterion, val_metric, NUM_CLASSES, DEVICE
        )
        
        print(f"\nValidation Results:")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val IoU:  {val_iou:.4f}")
    
    # Visualization
    if visualize_flag:
        print("\n" + "="*60)
        print("Generating Prediction Visualizations...")
        print("="*60)
        
        # Load best model
        checkpoint = torch.load(MODELS_DIR / "PSPNet_baseline.pt", weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Use validation set for visualization (test set labels are not public)
        save_predictions_grid(
            model, val_dataset, DEVICE,
            save_path=OUTPUTS_DIR / "PSPNet_baseline_predictions.png",
            num_samples=6
        )
        print(f"Predictions saved to {OUTPUTS_DIR / 'PSPNet_baseline_predictions.png'}")
    
    # Video Processing
    if process_videos_flag:
        print("\n" + "="*60)
        print("Processing Videos...")
        print("="*60)
        
        # Load best model
        checkpoint = torch.load(MODELS_DIR / "PSPNet_baseline.pt", weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        if VIDEOS_DIR.exists():
            # Get both .mp4 and .avi files
            video_files = list(VIDEOS_DIR.glob("*.mp4")) + list(VIDEOS_DIR.glob("*.avi"))
            for video_path in video_files:
                print(f"\nProcessing: {video_path.name}")
                # Keep the same extension as the input
                output_path = PROCESSED_DIR / f"{video_path.stem}_PSPNet_baseline_segmented{video_path.suffix}"
                # Pass the training input size so video frames are processed at same resolution
                process_video(model, video_path, output_path, DEVICE, input_size=(INPUT_HEIGHT, INPUT_WIDTH))
                print(f"Saved to: {output_path}")
        else:
            print(f"Videos directory not found: {VIDEOS_DIR}")
    
    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
