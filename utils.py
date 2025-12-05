"""
Utility Functions for PSPNet Segmentation
==========================================
This module contains all utility functions for:
- Dataset loading (Cityscapes)
- Training and validation loops
- Metrics (MeanIoU)
- Visualization
- Video processing
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from collections import namedtuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms


# ==================== Color Mapping ====================
# Cityscapes 19-class color mapping for visualization
Label = namedtuple("Label", ["name", "train_id", "color"])

cityscapes_labels = [
    Label("road", 0, (128, 64, 128)),
    Label("sidewalk", 1, (244, 35, 232)),
    Label("building", 2, (70, 70, 70)),
    Label("wall", 3, (102, 102, 156)),
    Label("fence", 4, (190, 153, 153)),
    Label("pole", 5, (153, 153, 153)),
    Label("traffic light", 6, (250, 170, 30)),
    Label("traffic sign", 7, (220, 220, 0)),
    Label("vegetation", 8, (107, 142, 35)),
    Label("terrain", 9, (152, 251, 152)),
    Label("sky", 10, (70, 130, 180)),
    Label("person", 11, (220, 20, 60)),
    Label("rider", 12, (255, 0, 0)),
    Label("car", 13, (0, 0, 142)),
    Label("truck", 14, (0, 0, 70)),
    Label("bus", 15, (0, 60, 100)),
    Label("train", 16, (0, 80, 100)),
    Label("motorcycle", 17, (0, 0, 230)),
    Label("bicycle", 18, (119, 11, 32)),
]

# Create ID to color mapping
train_id_to_color = {label.train_id: label.color for label in cityscapes_labels}

# Cityscapes labelId to trainId mapping (from official Cityscapes scripts)
# Maps from labelId (0-33) to trainId (0-18 or 255 for ignore)
LABEL_ID_TO_TRAIN_ID = np.array([
    255, 255, 255, 255, 255, 255, 255,  # 0-6: void classes
    0,   # 7: road
    1,   # 8: sidewalk
    255, # 9: parking (ignore)
    255, # 10: rail track (ignore)
    2,   # 11: building
    3,   # 12: wall
    4,   # 13: fence
    255, 255, 255,  # 14-16: more void
    5,   # 17: pole
    255, # 18: polegroup (ignore)
    6,   # 19: traffic light
    7,   # 20: traffic sign
    8,   # 21: vegetation
    9,   # 22: terrain
    10,  # 23: sky
    11,  # 24: person
    12,  # 25: rider
    13,  # 26: car
    14,  # 27: truck
    15,  # 28: bus
    255, # 29: caravan (ignore)
    255, # 30: trailer (ignore)
    16,  # 31: train
    17,  # 32: motorcycle
    18,  # 33: bicycle
], dtype=np.uint8)


# ==================== Image Preprocessing ====================
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

inverse_transform = transforms.Compose([
    transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))
])


# ==================== Dataset ====================
class CityscapesDataset(Dataset):
    """
    Cityscapes Dataset Loader
    
    Loads images and labels from PNG files (original PSPNet approach).
    Structure:
        data/dataset/
            leftImg8bit/
                train/*_leftImg8bit.png        # RGB images
                val/*_leftImg8bit.png
            gtFine/
                train/*_gtFine_labelIds.png    # Label masks
                val/*_gtFine_labelIds.png
    """
    
    def __init__(self, root_dir, split='train', crop_size=(512, 512), use_random_crop=True):
        """
        Args:
            root_dir: Path to dataset root (data/dataset)
            split: 'train' or 'val'
            crop_size: Crop size (H, W) - original paper uses 713x713
            use_random_crop: If True (training), use random crop. If False (val), use center crop.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.crop_size = crop_size
        self.use_random_crop = use_random_crop and (split == 'train')
        
        # Find all images
        self.image_files = []
        self.label_files = []
        
        # Images are in leftImg8bit/split/*_leftImg8bit.png
        images_dir = self.root_dir / "leftImg8bit" / split
        labels_dir = self.root_dir / "gtFine" / split
        
        if not images_dir.exists():
            raise ValueError(f"Images directory not found: {images_dir}")
        if not labels_dir.exists():
            raise ValueError(f"Labels directory not found: {labels_dir}")
        
        # Find all image files directly in the split directory
        for img_file in sorted(images_dir.glob("*_leftImg8bit.png")):
            # Corresponding label file (using labelIds, which we'll remap to trainIds)
            # e.g., aachen_000000_000019_leftImg8bit.png -> aachen_000000_000019_gtFine_labelIds.png
            base_name = img_file.stem.replace("_leftImg8bit", "")
            label_file = labels_dir / f"{base_name}_gtFine_labelIds.png"
            
            if label_file.exists():
                self.image_files.append(img_file)
                self.label_files.append(label_file)
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {images_dir}")
        
        print(f"{split.capitalize()} set: {len(self.image_files)} samples")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Tensor (3, H, W) - normalized
            label: Tensor (H, W) - class IDs
        """
        # Load image
        image = Image.open(self.image_files[idx]).convert('RGB')
        
        # Load label (labelIds format - needs remapping to trainIds)
        label = Image.open(self.label_files[idx])
        label = np.array(label, dtype=np.uint8)
        
        # Remap labelIds to trainIds using lookup table
        label = LABEL_ID_TO_TRAIN_ID[label]
        
        # Convert back to PIL for cropping
        label = Image.fromarray(label)
        
        # Random crop (training) or center crop (validation)
        # Cityscapes original size: 1024x2048
        if self.use_random_crop:
            # Random crop at original resolution (no resizing first!)
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=self.crop_size
            )
            image = transforms.functional.crop(image, i, j, h, w)
            label = transforms.functional.crop(label, i, j, h, w)
        else:
            # Center crop for validation
            image = transforms.functional.center_crop(image, self.crop_size)
            label = transforms.functional.center_crop(label, self.crop_size)
        
        # Convert to tensors
        image = preprocess(image)
        label = torch.from_numpy(np.array(label)).long()
        
        return image, label


# ==================== Metrics ====================
class MeanIoU:
    """
    Mean Intersection over Union (mIoU) metric for semantic segmentation.
    """
    
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset confusion matrix."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
    
    def update(self, preds, labels):
        """
        Update confusion matrix with predictions and labels.
        
        Args:
            preds: Tensor (B, C, H, W) - logits or (B, H, W) - class IDs
            labels: Tensor (B, H, W) - ground truth class IDs
        """
        # Handle different input formats
        if preds.dim() == 4:  # (B, C, H, W)
            preds = torch.argmax(preds, dim=1)  # (B, H, W)
        
        preds = preds.cpu().numpy().flatten()
        labels = labels.cpu().numpy().flatten()
        
        # Ignore pixels with label 255
        valid_mask = (labels != 255)
        preds = preds[valid_mask]
        labels = labels[valid_mask]
        
        # Update confusion matrix
        for pred, label in zip(preds, labels):
            if 0 <= label < self.num_classes and 0 <= pred < self.num_classes:
                self.confusion_matrix[label, pred] += 1
    
    def compute(self):
        """
        Compute mean IoU.
        
        Returns:
            mean_iou: Mean IoU across all classes
        """
        # IoU for each class
        intersection = np.diag(self.confusion_matrix)
        union = (
            self.confusion_matrix.sum(axis=1) +  # GT count
            self.confusion_matrix.sum(axis=0) -  # Pred count
            intersection  # Don't double count
        )
        
        # Avoid division by zero
        iou = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            if union[i] > 0:
                iou[i] = intersection[i] / union[i]
        
        # Mean IoU (only over classes that appear in GT)
        valid_classes = union > 0
        mean_iou = iou[valid_classes].mean() if valid_classes.sum() > 0 else 0.0
        
        return mean_iou


# ==================== Training & Validation ====================
def train_epoch(model, dataloader, criterion, optimizer, metric, num_classes, device, 
                scheduler=None, aux_weight=0.4):
    """
    Train for one epoch.
    
    Args:
        model: PSPNet model with auxiliary branch
        dataloader: Training data loader
        criterion: Loss function (CrossEntropyLoss)
        optimizer: Optimizer
        metric: MeanIoU metric
        num_classes: Number of classes
        device: Device to train on
        scheduler: Learning rate scheduler (optional)
        aux_weight: Weight for auxiliary loss
        
    Returns:
        avg_loss: Average loss for the epoch
        mean_iou: Mean IoU for the epoch
    """
    model.train()
    metric.reset()
    
    running_loss = 0.0
    running_main_loss = 0.0
    running_aux_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        # Handle auxiliary loss
        if isinstance(outputs, tuple):
            # Model returns (main_out, aux_out) during training
            main_out, aux_out = outputs
            main_loss = criterion(main_out, masks)
            aux_loss = criterion(aux_out, masks)
            loss = main_loss + aux_weight * aux_loss
            running_main_loss += main_loss.item()
            running_aux_loss += aux_loss.item()
        else:
            # Model returns only main output
            main_out = outputs
            loss = criterion(main_out, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Step scheduler per iteration (for polynomial decay)
        if scheduler is not None:
            scheduler.step()
        
        # Update metrics (use main output only)
        with torch.no_grad():
            metric.update(main_out, masks)
        
        running_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    
    avg_loss = running_loss / len(dataloader)
    mean_iou = metric.compute()
    
    return avg_loss, mean_iou


def validate_epoch(model, dataloader, criterion, metric, num_classes, device):
    """
    Validate for one epoch.
    
    Args:
        model: PSPNet model
        dataloader: Validation data loader
        criterion: Loss function
        metric: MeanIoU metric
        num_classes: Number of classes
        device: Device to validate on
        
    Returns:
        avg_loss: Average validation loss
        mean_iou: Mean IoU
    """
    model.eval()
    metric.reset()
    
    running_loss = 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass (model returns only main output in eval mode)
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, masks)
            running_loss += loss.item()
            
            # Update metrics
            metric.update(outputs, masks)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = running_loss / len(dataloader)
    mean_iou = metric.compute()
    
    return avg_loss, mean_iou


# ==================== Visualization ====================
def plot_training_curves(history, save_path):
    """
    Plot and save training curves.
    
    Args:
        history: Dict with 'train_loss', 'val_loss', 'train_iou', 'val_iou', 'lr'
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # IoU
    axes[0, 1].plot(epochs, history['train_iou'], 'b-', label='Train IoU')
    axes[0, 1].plot(epochs, history['val_iou'], 'r-', label='Val IoU')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Mean IoU')
    axes[0, 1].set_title('Training and Validation IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning Rate
    axes[1, 0].plot(epochs, history['lr'], 'g-')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].grid(True)
    axes[1, 0].set_yscale('log')
    
    # Summary text
    best_val_iou = max(history['val_iou'])
    best_epoch = history['val_iou'].index(best_val_iou) + 1
    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['val_loss'][-1]
    
    summary_text = f"""
    Training Summary:
    ─────────────────
    Best Val IoU: {best_val_iou:.4f} (Epoch {best_epoch})
    Final Train Loss: {final_train_loss:.4f}
    Final Val Loss: {final_val_loss:.4f}
    Total Epochs: {len(epochs)}
    """
    
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                    verticalalignment='center')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def decode_segmentation_mask(mask, id_to_color=train_id_to_color):
    """
    Convert class IDs to RGB color image.
    
    Args:
        mask: numpy array (H, W) with class IDs
        id_to_color: Dict mapping class ID to RGB color
        
    Returns:
        rgb: numpy array (H, W, 3) with RGB colors
    """
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Handle ignore class (255) - set to black
    for class_id, color in id_to_color.items():
        rgb[mask == class_id] = color
    
    # Pixels with class 255 (ignore) remain black (0, 0, 0)
    
    return rgb


def save_predictions_grid(model, dataset, device, save_path, num_samples=6):
    """
    Save a grid of prediction visualizations.
    
    Args:
        model: Trained model
        dataset: Test dataset
        device: Device
        save_path: Path to save image
        num_samples: Number of samples to visualize
    """
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    # Handle single row case
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i in range(num_samples):
            # Get sample
            image, label = dataset[i * (len(dataset) // num_samples)]
            
            # Predict
            image_input = image.unsqueeze(0).to(device)
            output = model(image_input)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            
            # Convert to RGB
            image_rgb = inverse_transform(image).permute(1, 2, 0).cpu().numpy()
            image_rgb = np.clip(image_rgb, 0, 1)
            
            # Convert labels to numpy and handle ignore class (255)
            label_np = label.numpy()
            # Create RGB for ground truth (set ignore pixels to black)
            label_rgb = decode_segmentation_mask(label_np)
            pred_rgb = decode_segmentation_mask(pred)
            
            # Plot
            axes[i, 0].imshow(image_rgb)
            axes[i, 0].set_title('Input Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(label_rgb)
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred_rgb)
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ==================== Video Processing ====================
def process_video(model, video_path, output_path, device, input_size=(720, 720)):
    """
    Process video with side-by-side segmentation view.
    
    Creates output with original frame on left and segmented frame on right.
    
    1. Downsizes each frame to model input size
    2. Runs segmentation inference
    3. Upsizes prediction back to original frame resolution
    4. Places original and segmented frames side-by-side
    
    Args:
        model: Trained model
        video_path: Input video path
        output_path: Output video path
        device: Device
        input_size: Model input size (H, W)
    """
    model.eval()
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  Video: {orig_width}x{orig_height} @ {fps} FPS, {total_frames} frames")
    print(f"  Model input: {input_size[1]}x{input_size[0]}")
    print(f"  Output: {orig_width*2}x{orig_height} (side-by-side)")
    
    # Video writer - output at double width for side-by-side
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (orig_width * 2, orig_height))
    
    pbar = tqdm(total=total_frames, desc="Processing video")
    
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Step 1: Downsize frame to model input size
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # Resize to model input size
            frame_resized = frame_pil.resize((input_size[1], input_size[0]), Image.BILINEAR)
            
            # Preprocess (normalize only, no resize)
            frame_tensor = preprocess(frame_resized).unsqueeze(0).to(device)
            
            # Step 2: Run inference
            output = model(frame_tensor)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            
            # Step 3: Upsize prediction back to ORIGINAL frame size
            pred_resized = cv2.resize(
                pred.astype(np.uint8), (orig_width, orig_height), 
                interpolation=cv2.INTER_NEAREST
            )
            
            # Convert prediction to RGB
            seg_rgb = decode_segmentation_mask(pred_resized)
            seg_bgr = cv2.cvtColor(seg_rgb, cv2.COLOR_RGB2BGR)
            
            # Step 4: Create side-by-side view (original | segmented)
            side_by_side = np.hstack([frame, seg_bgr])
            
            # Write frame at double width
            out.write(side_by_side)
            pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    print(f"  [DONE] Output video saved at {orig_width*2}x{orig_height} (side-by-side)")
