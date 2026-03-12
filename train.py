from vired_model.models.vired_model import ViREDModel
import os, yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader
import warnings
import cv2, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.amp.autocast_mode import autocast
from utils.eval import pair_metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore", category=FutureWarning)

class CrossEntropyLossWithMask(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(**kwargs, reduction='none')
    def forward(
            self, 
            logits: torch.Tensor, 
            labels: torch.Tensor, 
            padding_mask: torch.Tensor
        ):
        """
        logits: (B, P_max, C), logit for each class
        labels: (B, P_max), class indices
        padding_mask: (B, P_max), True if is padding
        """
        loss_unmasked = self.cross_entropy(logits, labels) # (B, P_max)
        mask = (~padding_mask).to(loss_unmasked.dtype)  
        loss_unreduced = loss_unmasked * mask
        loss = loss_unreduced.sum()
        return loss


class ViREDTrainer:
    def __init__(
        self,
        model: ViREDModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: str = 'cuda'
    ):
        """
        Args:
            model: Model for circuitry segmentation
            train_loader: DataLoader for training (returns dicts with image/mask pairs)
            val_loader: DataLoader for validation
            device: Device to train on
            config: Configuration dictionary
        """
        self.config = config
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
   
        lr = self.config.get('lr', 1e-4)
        weight_decay = self.config.get('weight_decay', 1e-5)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.config.get('scheduler_patience', 5),
            factor=self.config.get('scheduler_factor', 0.1)
        )
        self.criterion = CrossEntropyLossWithMask()
        self.save_path = self.config.get('save_path', 'checkpoints')
        os.makedirs(self.save_path, exist_ok=True)
        with open(os.path.join(self.save_path, 'train_config.yaml'), 'w') as f:
            yaml.dump(self.config, f)
        self.autocast_enabled = self.config.get('autocast_enabled', True)
        
        self.best_val_loss = float('inf')
        self.validate = self.config.get('validate', True)
        self.start_validate = self.config.get('start_validate', False)
        
        self.save_checkpoint_interval = self.config.get('save_checkpoint_interval', 1)
        self.loss_log_interval = self.config.get('loss_log_interval', 100)
        
        self.grad_clip = self.config.get('grad_clip', None)

    
        
        print(f"ViREDTrainer initialized | Device: {device} | Learning rate: {lr} | Weight decay: {weight_decay}")

    def train_one_epoch(self, epoch: int = 0) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f'Training Epoch {epoch+1}')
        iter_count = 0

        epoch_scores = []
        epoch_labels = []

        
        for i, batch in enumerate(pbar, 1):
            self.optimizer.zero_grad()
            image = batch['images'].to(self.device)
            object_masks = batch['object_masks'].to(self.device)
            object_boxes = batch['object_boxes'].to(self.device)
            object_types = batch['object_types'].to(self.device)
            object_key_padding_mask = batch['object_key_padding_mask'].to(self.device)
            pair_indices_gt = batch['pair_indices'].to(self.device)
            pair_labels_gt = batch['pair_labels'].to(self.device)
            pair_padding_mask_gt = batch['pair_padding_mask'].to(self.device)
            
            with autocast(str(self.device), enabled=self.autocast_enabled):
                output = self.model(
                            image, 
                            object_masks, 
                            object_boxes, 
                            object_types,
                            object_key_padding_mask
                        )
                pair_logits = output["pair_logits"]
                pair_indices_pred = output["pair_indices"]
                pair_padding_mask_pred = output["pair_padding_mask"]
                assert torch.equal(pair_indices_gt, pair_indices_pred), f"pair_indices_gt and pair_indices_pred are not equal. pair_indices_gt: {pair_indices_gt.shape} pair_indices_pred: {pair_indices_pred.shape}"
                assert torch.equal(pair_padding_mask_gt, pair_padding_mask_pred), f"pair_padding_mask_gt and pair_padding_mask_pred are not equal. pair_padding_mask_gt: {pair_padding_mask_gt.shape} pair_padding_mask_pred: {pair_padding_mask_pred.shape}"
                loss = self.criterion(logits=pair_logits, labels=pair_labels_gt, padding_mask=pair_padding_mask_pred)


            if torch.isnan(loss):
                print(f"[NaN Detected] Step: {i}")
                print(f"image stats: min={image.min().item():.4f}, max={image.max().item():.4f}, mean={image.mean().item():.4f}")
                print(f"pair_logits stats: min={pair_logits.min().item():.4f}, max={pair_logits.max().item():.4f}, mean={pair_logits.mean().item():.4f}")
                print(f"pair_padding_mask_pred stats: min={pair_padding_mask_pred.min().item():.4f}, max={pair_padding_mask_pred.max().item():.4f}, mean={pair_padding_mask_pred.mean().item():.4f}")
                print("Skipping this batch")
                del image, pair_logits, pair_padding_mask_pred, loss
                torch.cuda.empty_cache() if self.device == 'cuda' else None
                continue
            loss.backward()

            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            iter_count += 1
            avg_loss = total_loss / iter_count
            
            with torch.no_grad():
                probs = torch.softmax(pair_logits, dim=-1)[..., 1]

                valid_mask = ~pair_padding_mask_pred

                epoch_scores.append(probs[valid_mask].detach().cpu())
                epoch_labels.append(pair_labels_gt[valid_mask].detach().cpu())
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{avg_loss:.4f}",
            })
        
        epoch_scores = torch.cat(epoch_scores)
        epoch_labels = torch.cat(epoch_labels)
        preds = (epoch_scores >= 0.5).int()

        metrics = {
            "accuracy": accuracy_score(epoch_labels, preds),
            "precision": precision_score(epoch_labels, preds, zero_division=0), # type: ignore
            "recall": recall_score(epoch_labels, preds, zero_division=0), # type: ignore
            "f1": f1_score(epoch_labels, preds, zero_division=0), # type: ignore
            "ap": average_precision_score(epoch_labels, epoch_scores),
            "auroc": roc_auc_score(epoch_labels, epoch_scores),
        }
        return {
            'loss': avg_loss,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "ap": metrics["ap"],
            "auroc": metrics["auroc"]
        }

    def validate_one_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        pbar = tqdm(self.val_loader, desc='Validation')
        iter_count = 0
        
        with torch.no_grad():
            for i, batch in enumerate(pbar, 1):
                image = batch['image'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                with autocast(str(self.device), enabled=self.autocast_enabled):
                    pred = self.model(image)
                    loss = self.criterion(pred, mask.unsqueeze(1).float()) if self.config.get('criterion', 'BCEWithLogitsLoss') == 'BCEWithLogitsLoss' else self.criterion(pred, mask)
                
                pred_probs = torch.sigmoid(pred) if self.config.get('criterion', 'BCEWithLogitsLoss') == 'BCEWithLogitsLoss' else torch.softmax(pred, dim=1)[:, 1:2]
                metrics = metrics_interactive(pred_probs, mask.unsqueeze(1).float())
                
                total_loss += loss.item()
                total_dice += metrics['dice'].item()
                total_iou += metrics['iou'].item()
                iter_count += 1
                avg_loss = total_loss / iter_count
                avg_dice = total_dice / iter_count
                avg_iou = total_iou / iter_count
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg_loss': f"{avg_loss:.4f}",
                    'dice': f"{avg_dice:.4f}",
                    'iou': f"{avg_iou:.4f}",
                })
        
        return {
            'loss': avg_loss,
            'dice': avg_dice,
            'iou': avg_iou
        }

    def run(self, epochs: int = 50):
        """Run training loop."""
        train_losses = []
        train_dices = []
        train_ious = []
        val_losses = []
        val_dices = []
        val_ious = []
        val_loss = float('inf')
        if self.start_validate:
            print("Running initial validation...")
            val_metrics = self.validate_one_epoch()
            val_loss = val_metrics['loss']
            print(f"Initial Val Loss: {val_loss:.4f} | Dice: {val_metrics['dice']:.4f} | IoU: {val_metrics['iou']:.4f}")
            val_losses.append(val_loss)
            val_dices.append(val_metrics['dice'])
            val_ious.append(val_metrics['iou'])
        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"{'='*60}")
            train_metrics = self.train_one_epoch(epoch)
            train_loss = train_metrics['loss']
            train_losses.append(train_loss)
            train_dices.append(train_metrics['dice'])
            train_ious.append(train_metrics['iou'])
            print(f"Train Loss: {train_loss:.4f} | Dice: {train_metrics['dice']:.4f} | IoU: {train_metrics['iou']:.4f}")
            if self.validate:
                val_metrics = self.validate_one_epoch()
                val_loss = val_metrics['loss']
                val_losses.append(val_loss)
                val_dices.append(val_metrics['dice'])
                val_ious.append(val_metrics['iou'])
                print(f"Val Loss: {val_loss:.4f} | Dice: {val_metrics['dice']:.4f} | IoU: {val_metrics['iou']:.4f}")
                self.scheduler.step(val_loss)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_path = os.path.join(
                        self.save_path,
                        f'best_val_{val_loss:.4f}_train_{train_loss:.4f}.pth'
                    )
                    torch.save(self.model.state_dict(), best_path)
                    print(f"✓ Saved best model: {best_path}")
                if (epoch + 1) % self.save_checkpoint_interval == 0 and not val_loss < self.best_val_loss:
                    checkpoint_path = os.path.join(
                        self.save_path,
                        f'checkpoint_epoch_{epoch+1}_val_{val_loss:.4f}.pth'
                    )
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'val_loss': val_loss,
                        'train_loss': train_loss,
                        'best_val_loss': self.best_val_loss,
                    }, checkpoint_path)
                    print(f"✓ Saved checkpoint: {checkpoint_path}")
            else:
                self.scheduler.step(train_loss)
                val_losses.append(float('inf'))
                val_dices.append(0.0)
                val_ious.append(0.0)
                if (epoch + 1) % 5 == 0:
                    checkpoint_path = os.path.join(
                        self.save_path,
                        f'checkpoint_epoch_{epoch+1}_val_{val_loss:.4f}.pth'
                    )
                    torch.save(self.model.state_dict(), checkpoint_path)
                    print(f"✓ Saved checkpoint: {checkpoint_path}")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        epochs_range = range(1, len(train_losses) + 1)
        axes[0].plot(epochs_range, train_losses, 'b-', label='Training Loss', linewidth=2)
        if self.validate and len(val_losses) > 0:
            if self.start_validate:
                val_epochs = [0] + list(range(1, len(val_losses)))
            else:
                val_epochs = range(1, len(val_losses) + 1)
            axes[0].plot(val_epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(epochs_range, train_dices, 'b-', label='Training Dice', linewidth=2)
        if self.validate and len(val_dices) > 0:
            if self.start_validate:
                val_epochs = [0] + list(range(1, len(val_dices)))
            else:
                val_epochs = range(1, len(val_dices) + 1)
            axes[1].plot(val_epochs, val_dices, 'r-', label='Validation Dice', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Dice Coefficient', fontsize=12)
        axes[1].set_title('Training and Validation Dice', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(epochs_range, train_ious, 'b-', label='Training IoU', linewidth=2)
        if self.validate and len(val_ious) > 0:
            if self.start_validate:
                val_epochs = [0] + list(range(1, len(val_ious)))
            else:
                val_epochs = range(1, len(val_ious) + 1)
            axes[2].plot(val_epochs, val_ious, 'r-', label='Validation IoU', linewidth=2)
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('IoU', fontsize=12)
        axes[2].set_title('Training and Validation IoU', fontsize=14, fontweight='bold')
        axes[2].legend(fontsize=11)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.save_path, 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved training curves plot: {plot_path}")
        
        print(f"\n{'='*60}")
        print("Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}")
