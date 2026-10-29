"""MC-dropout inference: keep BatchNorm in eval mode, enable only dropout."""
import torch
import torch.nn as nn

_DROPOUT_TYPES = (nn.Dropout, nn.Dropout2d, nn.Dropout3d)

def enable_mc_dropout(model):
    model.eval()
    for module in model.modules():
        if isinstance(module, _DROPOUT_TYPES):
            module.train()
    return model

@torch.no_grad()
def mc_predict(model, images, passes=10):
    if passes < 2:
        raise ValueError("Use at least 2 MC passes to estimate uncertainty.")
    enable_mc_dropout(model)
    samples = []
    for _ in range(passes):
        samples.append(torch.softmax(model(images), dim=1))
    probs = torch.stack(samples, dim=0)  # (T,B,C,H,W)
    variance = probs.var(dim=0, unbiased=False)
    return probs, variance
