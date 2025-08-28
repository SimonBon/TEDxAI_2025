import os
import math
from typing import Union, List, Literal, Optional
from pathlib import Path

from warnings import warn
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from mmcv.transforms import Compose
from mmselfsup.datasets.transforms import PackSelfSupInputs

from mmselfsup.registry import DATASETS

@DATASETS.register_module()
class PatchDataset(Dataset):

    def __init__(self,
                h5_file: Union[Path, str],
                mode: Union[str, Literal["classification", "regression"]],
                pipeline: List[List[dict]] = None,
                **kwargs):
        
        super().__init__()
           
        # Ensure the HDF5 file exists
        assert Path(h5_file).exists(), f"Provided path to h5 file does not exist: {h5_file}"
        self.h5_file = h5py.File(h5_file, 'r')
        print(f"H5-File: {h5_file}")
        print(f"H5-File has the follwing keys: {self.h5_file.keys()}")
                        
        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        else:
            self.pipeline = Compose([PackSelfSupInputs(algorithm_keys=["gt_label"])])
          
        assert mode in ['classification', 'regression'], 'Invalid "mode" chose one of "classification" or "regression"'  
        self.mode = mode
            
            
    def __len__(self):
        return len(self.h5_file['image_patches'])


    def __getitem__(self, idx, eval=False, ommit_pipeline=False):
        
        img = self.h5_file['image_patches'][idx]
        mask = torch.tensor(self.h5_file['mask_patches'][idx])
        # size = self.h5_file['sizes'][idx]
        gt_label_class = torch.tensor([self.h5_file['class'][idx]])
        
        if self.mode == 'regression':
            parameters = self.h5_file['parameters'][idx]
            gt_label_spots = torch.tensor([float(parameters[3] + parameters[4] * parameters[5])])
        else:
            gt_label_spots = torch.tensor([0])
            
        # idx = self.row_idxs[idx] # if not shuffle: eg orig_idx:134 -> idx:134 | if shuffle eg orig_idx:134 -> idx:829
        
        # img_idx, x, y, _, _ = self.h5_file["INDEXER"][idx].astype(int)
                  
        # # Extract and pad patches and masks
        # current_patch = self.extract_and_pad(img_idx, x, y)

        # current_patch = current_patch[..., self.used_channels]

        pipeline_dict = dict(
            img = img,
            mask = mask,
            # size = size,
            gt_label_class = gt_label_class,
            gt_label_spots = gt_label_spots
            #parameters = parameters
        )
                                                            
        return self.pipeline(pipeline_dict)