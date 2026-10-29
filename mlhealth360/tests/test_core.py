import numpy as np
import torch
from src.data.hu import hu_to_window
from src.data.windowing import window_slice
from src.models.unet import UNet25D, count_dropout
from src.training.loss import combined_loss
from src.active_learning.mc_dropout import mc_predict

def test_window_range():
    x=np.array([[-1000,40,1000]],dtype=np.float32)
    y=hu_to_window(x,40,80)
    assert y.min() >= 0 and y.max() <= 1

def test_three_windows():
    y=window_slice(np.zeros((32,32),dtype=np.float32))
    assert y.shape==(3,32,32)

def test_model_shape_dropout_and_loss():
    model=UNet25D(in_channels=9,num_classes=6,base_channels=4,dropout_p=.2)
    assert count_dropout(model)>=1
    x=torch.randn(2,9,64,64)
    y=torch.randint(0,6,(2,64,64))
    logits=model(x)
    assert logits.shape==(2,6,64,64)
    loss,_=combined_loss(logits,y)
    assert torch.isfinite(loss)

def test_mc_dropout_is_stochastic():
    torch.manual_seed(7)
    model=UNet25D(in_channels=9,num_classes=6,base_channels=4,dropout_p=.5)
    x=torch.randn(1,9,64,64)
    probs,_=mc_predict(model,x,passes=3)
    assert probs.shape==(3,1,6,64,64)
    assert probs.var(0).mean().item()>0
