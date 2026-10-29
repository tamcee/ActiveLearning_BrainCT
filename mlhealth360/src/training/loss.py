"""Cross-entropy + foreground multiclass soft Dice loss."""
import torch
import torch.nn.functional as F

def multiclass_dice_loss(logits, targets, num_classes=6, eps=1e-6):
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = (probs * one_hot).sum(dim=dims)
    denominator = probs.sum(dim=dims) + one_hot.sum(dim=dims)
    dice = (2 * intersection + eps) / (denominator + eps)
    return 1.0 - dice[1:].mean()  # ignore background in Dice term

def combined_loss(logits, targets, dice_weight=1.0, num_classes=6):
    ce = F.cross_entropy(logits, targets.long())
    dice = multiclass_dice_loss(logits, targets, num_classes=num_classes)
    total = ce + float(dice_weight) * dice
    return total, {
        "cross_entropy": float(ce.detach()),
        "dice_loss": float(dice.detach()),
        "total_loss": float(total.detach()),
    }
