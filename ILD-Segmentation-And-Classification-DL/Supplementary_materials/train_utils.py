import os
import time
import torch
import numpy as np
from sklearn.metrics import classification_report

class Logger:
    def __init__(self, exp_name, root_dir="experiments"):
        self.exp_dir = os.path.join(root_dir, f"{exp_name}_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.exp_dir, exist_ok=True)
        self.log_file = open(os.path.join(self.exp_dir, "train.log"), "w")
        self.ckpt_dir = os.path.join(self.exp_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def log(self, message):
        print(message)
        self.log_file.write(message + "\n")
        self.log_file.flush()

    def save_ckpt(self, model, optimizer, epoch, is_best=False):
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        filename = os.path.join(self.ckpt_dir, f"epoch_{epoch:03d}.pth")
        torch.save(state, filename)
        if is_best:
            torch.save(state, os.path.join(self.ckpt_dir, "best_model.pth"))

    def close(self):
        self.log_file.close()

class MetricTracker:
    def __init__(self, num_classes, task_type='classification'):
        self.num_classes = num_classes
        self.task_type = task_type
        self.reset()

    def reset(self):
        self.y_true = []
        self.y_pred = []

    def update(self, y_true, y_pred):
        if self.task_type == 'classification':
            # y_true, y_pred are (B,) tensors
            self.y_true.extend(y_true.cpu().numpy().tolist())
            self.y_pred.extend(y_pred.cpu().numpy().tolist())
        elif self.task_type == 'segmentation':
            # y_true: (B, H, W) long, y_pred: (B, C, H, W) logits
            pred_labels = torch.argmax(y_pred, dim=1) # (B, H, W)
            self.y_true.append(y_true.cpu().numpy())
            self.y_pred.append(pred_labels.cpu().numpy())

    def get_report(self, class_names=None):
        if self.task_type == 'classification':
            return classification_report(self.y_true, self.y_pred, target_names=class_names, digits=4, zero_division=0)
        else:
            # Segmentation: Calculate Dice per class
            y_true_all = np.concatenate(self.y_true, axis=0) # (N, H, W)
            y_pred_all = np.concatenate(self.y_pred, axis=0) # (N, H, W)
            
            dice_scores = []
            report = "Per-class Dice Scores:\n"
            for i in range(self.num_classes):
                intersect = np.sum((y_true_all == i) & (y_pred_all == i))
                union = np.sum(y_true_all == i) + np.sum(y_pred_all == i)
                dice = (2. * intersect) / (union + 1e-6)
                dice_scores.append(dice)
                name = class_names[i] if class_names else f"Class {i}"
                report += f"  {name}: {dice:.4f}\n"
            
            avg_dice = np.mean(dice_scores)
            report += f"Mean Dice: {avg_dice:.4f}"
            return report
