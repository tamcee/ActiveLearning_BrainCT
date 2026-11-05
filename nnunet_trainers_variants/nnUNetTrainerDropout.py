import os, sys, time
import numpy as np
import torch
import torch.nn as nn
from typing import Union, List, Tuple
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

def count_dropout_layers(module: nn.Module) -> int:
    return sum(1 for m in module.modules() if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)))

class SafeClassLocations(dict):
    def __init__(self, original_dict, patch_dim=2):
        super().__init__(original_dict if original_dict is not None else {})
        self.patch_dim = patch_dim

    def _has_c(self, c):
        try:
            return super().__contains__(c) or super().__contains__(int(c))
        except (ValueError, TypeError):
            return super().__contains__(c)

    def _get_c(self, c):
        if super().__contains__(c):
            return super().__getitem__(c)
        try:
            if super().__contains__(int(c)):
                return super().__getitem__(int(c))
        except (ValueError, TypeError):
            pass
        return None

    def __getitem__(self, key):
        if self._has_c(key):
            val = self._get_c(key)
            if val is not None:
                return val
        if isinstance(key, (tuple, list, np.ndarray)):
            fg_locs = []
            for c in key:
                try:
                    c_int = int(c)
                except (ValueError, TypeError):
                    c_int = c
                if c_int > 0 and c_int != 6 and self._has_c(c):
                    v = self._get_c(c)
                    if v is not None and len(v) > 0:
                        fg_locs.append(v)
            if len(fg_locs) > 0:
                return np.concatenate(fg_locs, axis=0)
            any_locs = []
            for c in key:
                if self._has_c(c):
                    v = self._get_c(c)
                    if v is not None and len(v) > 0:
                        any_locs.append(v)
            if len(any_locs) > 0:
                return np.concatenate(any_locs, axis=0)
            return np.zeros((0, self.patch_dim), dtype=int)
        return np.zeros((0, self.patch_dim), dtype=int)

    def __contains__(self, key):
        return True

try:
    from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
    _orig_get_bbox = nnUNetDataLoader.get_bbox
    def _safe_get_bbox(self, data_shape, force_fg, class_locations):
        patch_dim = len(self.patch_size) if hasattr(self, 'patch_size') else 2
        safe_locs = SafeClassLocations(class_locations, patch_dim=patch_dim) if class_locations is not None else None
        return _orig_get_bbox(self, data_shape, force_fg, safe_locs)
    nnUNetDataLoader.get_bbox = _safe_get_bbox
except Exception:
    pass

try:
    import nnunetv2.training.dataloading.data_loader_2d as dl2d
    if hasattr(dl2d, 'nnUNetDataLoader2D') and hasattr(dl2d.nnUNetDataLoader2D, 'get_bbox'):
        _orig_get_bbox_2d = dl2d.nnUNetDataLoader2D.get_bbox
        def _safe_get_bbox_2d(self, data_shape, force_fg, class_locations):
            patch_dim = len(self.patch_size) if hasattr(self, 'patch_size') else 2
            safe_locs = SafeClassLocations(class_locations, patch_dim=patch_dim) if class_locations is not None else None
            return _orig_get_bbox_2d(self, data_shape, force_fg, safe_locs)
        dl2d.nnUNetDataLoader2D.get_bbox = _safe_get_bbox_2d
except Exception:
    pass

class nnUNetTrainerDropout(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device=None):
        if device is None:
            super().__init__(plans, configuration, fold, dataset_json)
        else:
            super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.num_epochs = 50
        self.save_every = 1
        self.best_macro_dice = -float('inf')

    def get_dataloaders(self):
        if os.name == 'nt':
            os.environ['nnUNet_n_proc_DA'] = '0'
        return super().get_dataloaders()

    def initialize(self):
        super().initialize()
        if hasattr(self.network, 'decoder'):
            dec = self.network.decoder
            if hasattr(dec, 'stages'):
                for s_idx in range(len(dec.stages)):
                    dec.stages[s_idx] = nn.Sequential(dec.stages[s_idx], nn.Dropout2d(p=0.2)).to(self.device)
            else:
                for name, child in list(dec.named_children()):
                    if isinstance(child, nn.Sequential):
                        child.add_module('mc_dropout2d', nn.Dropout2d(p=0.2))
        if count_dropout_layers(self.network) == 0 and hasattr(self.network, 'seg_layers'):
            new_segs = nn.ModuleList([nn.Sequential(nn.Dropout2d(p=0.2), s).to(self.device) for s in self.network.seg_layers])
            self.network.seg_layers = new_segs
        num_drop_verified = count_dropout_layers(self.network)
        assert num_drop_verified >= 1, f'Fatal: Dropout count is {num_drop_verified} < 1! MC-Dropout cannot function.'
        print(f'[nnUNetTrainerDropout] Startup assertion PASSED: {num_drop_verified} Dropout layers active (p>=0.2).')

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self._iter_count = 0
        self._epoch_start_time = time.time()

    def train_step(self, batch: dict) -> dict:
        out = super().train_step(batch)
        self._iter_count += 1
        if self._iter_count % 25 == 0 or self._iter_count == self.num_iterations_per_epoch:
            elapsed = time.time() - self._epoch_start_time
            speed = self._iter_count / max(1e-5, elapsed)
            loss_val = float(out.get('loss', 0.0))
            print(f'  [Epoch {self.current_epoch}] Step {self._iter_count}/{self.num_iterations_per_epoch} ({self._iter_count*100//self.num_iterations_per_epoch}%) - Loss: {loss_val:.4f} ({speed:.2f} step/s)', flush=True)
        return out

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self._val_iter_count = 0

    def validation_step(self, batch: dict) -> dict:
        out = super().validation_step(batch)
        self._val_iter_count += 1
        if self._val_iter_count % 10 == 0 or self._val_iter_count == self.num_val_iterations_per_epoch:
            print(f'  [Epoch {self.current_epoch} Val] Step {self._val_iter_count}/{self.num_val_iterations_per_epoch}...', flush=True)
        return out

    def on_epoch_end(self):
        super().on_epoch_end()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            log = getattr(self.logger, 'my_fantastic_logging', {})
            values = log.get('mean_fg_dice', []) if isinstance(log, dict) else []
            if len(values):
                current_fg_dice = float(np.asarray(values[-1]).reshape(-1)[0])
                if current_fg_dice > self.best_macro_dice:
                    self.best_macro_dice = current_fg_dice
                    out_path = os.path.join(self.output_folder, 'checkpoint_best_macro_dice.pth') if hasattr(self, 'output_folder') else 'checkpoint_best_macro_dice.pth'
                    self.save_checkpoint(out_path)
                    print(f'[*] New best validation macro Dice={current_fg_dice:.5f} saved at epoch {self.current_epoch}')
        except Exception as e:
            pass

class nnUNetTrainerALFineTune(nnUNetTrainerDropout):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device=None):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.num_epochs = 25
        self.initial_lr = 1e-5
