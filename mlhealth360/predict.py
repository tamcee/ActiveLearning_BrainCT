"""Run a trained 2.5D U-Net on a single NIfTI volume and save slice predictions."""
import argparse
from pathlib import Path
import numpy as np, nibabel as nib
import torch, yaml
from src.data.windowing import window_slice
from src.models.unet import UNet25D
from src.training.checkpoint import load_checkpoint

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--image",required=True)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--model-config",default="configs/model.yaml")
    p.add_argument("--output",default="prediction_mask.nii.gz")
    args=p.parse_args()
    cfg=yaml.safe_load(open(args.model_config,encoding="utf-8"))["model"]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=UNet25D(**{k:cfg[k] for k in ("in_channels","num_classes","base_channels","dropout_p")}).to(device)
    load_checkpoint(args.checkpoint,model,map_location=device);model.eval()
    nii=nib.load(args.image);vol=np.asarray(nii.dataobj,dtype=np.float32);depth=vol.shape[2]
    pred=np.zeros(vol.shape,dtype=np.uint8)
    with torch.no_grad():
        for z in range(depth):
            slices=[window_slice(vol[:,:,min(max(z+d,0),depth-1)]) for d in (-1,0,1)]
            x=torch.from_numpy(np.concatenate(slices,axis=0)).float().unsqueeze(0).to(device)
            pred[:,:,z]=model(x).argmax(1)[0].cpu().numpy().astype(np.uint8)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    nib.save(nib.Nifti1Image(pred,nii.affine,nii.header),str(out))
    print("Saved",out)

if __name__=="__main__":main()
