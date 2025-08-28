# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch

from mmselfsup.registry import MODELS
from mmselfsup.structures import SelfSupDataSample
from mmselfsup.models.utils import GatherLayer
from mmselfsup.models.algorithms.base import BaseModel
from mmengine.structures.label_data import LabelData

from mmengine.registry import build_model_from_cfg

import warnings
warnings.simplefilter("once", UserWarning)


@MODELS.register_module()
class SimCLRPlusClassifier(BaseModel):
    """SimCLR.

    Implementation of `A Simple Framework for Contrastive Learning of Visual
    Representations <https://arxiv.org/abs/2002.05709>`_.
    """
    def __init__(self, classifier=None, regressor=None, reducer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
        self.reducer = build_model_from_cfg(reducer, MODELS) if reducer is not None else None
        self.classifier = build_model_from_cfg(classifier, MODELS) if classifier is not None else None
        self.regressor = build_model_from_cfg(regressor, MODELS) if regressor is not None else None


    @staticmethod
    def _create_buffer(
        batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the mask and the index of positive samples.

        Args:
            batch_size (int): The batch size.
            device (torch.device): The device of backend.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - The mask for feature selection.
            - The index of positive samples.
            - The mask of negative samples.
        """
        mask = 1 - torch.eye(batch_size * 2, dtype=torch.uint8).to(device)
        pos_idx = (
            torch.arange(batch_size * 2).to(device),
            2 * torch.arange(batch_size, dtype=torch.long).unsqueeze(1).repeat(
                1, 2).view(-1, 1).squeeze().to(device))
        neg_mask = torch.ones((batch_size * 2, batch_size * 2 - 1),
                              dtype=torch.uint8).to(device)
        neg_mask[pos_idx] = 0
        return mask, pos_idx, neg_mask

    def extract_feat(self, inputs: List[torch.Tensor],
                     **kwargs) -> Tuple[torch.Tensor]:
        """Function to extract features from backbone.

        Args:
            inputs (List[torch.Tensor]): The input images.

        Returns:
            Tuple[torch.Tensor]: Backbone outputs.
        """
        features = self.backbone(inputs[0])
        features = self.reducer(features)
        
        return features

    def loss(self, inputs: List[torch.Tensor],
             data_samples: List[SelfSupDataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        """The forward function in training.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """
        
        device = next(self.backbone.parameters()).device
        
        assert isinstance(inputs, list)
        inputs = torch.stack(inputs, 1)
        inputs = inputs.reshape((inputs.size(0) * 2, inputs.size(2),
                                 inputs.size(3), inputs.size(4)))
        features = self.backbone(inputs)
        if self.reducer is not None:
            features = self.reducer(features)
            
        z = self.neck(features)[0]  # (2n)xd

        z = z / (torch.norm(z, p=2, dim=1, keepdim=True) + 1e-10)
        z = torch.cat(GatherLayer.apply(z), dim=0)  # (2N)xd
        
        assert z.size(0) % 2 == 0
        
        N = z.size(0) // 2
        s = torch.matmul(z, z.permute(1, 0))  # (2N)features(2N)
        mask, pos_idx, neg_mask = self._create_buffer(N, s.device)

        # remove diagonal, (2N)features(2N-1)
        s = torch.masked_select(s, mask == 1).reshape(s.size(0), -1)
        positive = s[pos_idx].unsqueeze(1)  # (2N)x1

        # select negative, (2N)features(2N-2)
        negative = torch.masked_select(s, neg_mask == 1).reshape(s.size(0), -1)

        contrastive = self.head(positive, negative)
        
        if self.classifier is not None:
        
            two_view_data_samples = []
            for k in data_samples:
                two_view_data_samples.append(k)
                two_view_data_samples.append(k)
                
            for k in two_view_data_samples:
                setattr(k, 'gt_label', LabelData())
                setattr(k.gt_label, 'label', torch.tensor([k.pseudo_label.gt_label_class]).to(device))
            
            classifier = self.classifier.loss(features, two_view_data_samples)['loss']
            
        else:
            
            classifier = torch.tensor(0.).to(device)
            
            
        if self.regressor is not None:
        
            two_view_data_samples = []
            for k in data_samples:
                two_view_data_samples.append(k)
                two_view_data_samples.append(k)
                
            for k in two_view_data_samples:
                setattr(k, 'gt_label', LabelData())
                setattr(k.gt_label, 'label', torch.tensor([k.pseudo_label.gt_label_spots]).to(device))
            
            regressor = self.regressor.loss(features, two_view_data_samples)['loss']
            
        else:
            
            regressor = torch.tensor(0.).to(device)
            
        
        # Weighted total (single trainable key must be named 'loss')
        total = contrastive + classifier + regressor

        # Log components under names that WON'T be summed
        return dict(
            loss=total,
            contrastive=contrastive.detach(),
            classifier=classifier.detach(),
            regressor=regressor.detach()
        )
    

    def predict(
        self, 
        inputs, 
        data_samples=None, 
        return_features=False, 
        return_images=False, 
        return_uncertainty=False,
        regression=False):

        device = next(self.backbone.parameters()).device
                
        if data_samples is not None:
            assert len(inputs[0]) == len(data_samples)

        # forward
        features = self.backbone(inputs[0].to(device))
        if self.reducer is not None:
            features = self.reducer(features)
        class_logits = self.classifier(features)            
        regression = self.regressor(features)

        # ----- uncertainty from class_logits -----
        if return_uncertainty:
            with torch.no_grad():
                probs = torch.softmax(class_logits, dim=1)            
                log_probs = torch.log_softmax(class_logits, dim=1)
                entropy = -(probs * log_probs).sum(dim=1)       
                C = probs.size(1)
                norm_const = torch.log(torch.tensor(C, device=class_logits.device, dtype=class_logits.dtype))
                normalized_entropy = entropy / norm_const        

        # hard predictions
        classification = torch.argmax(class_logits, dim=1)          

        # ----- ground-truth labels -----
        if data_samples is None:
            gt_label_class = torch.full_like(classification, fill_value=-1)
            gt_label_spots = torch.full_like(classification, fill_value=-1)
            
        else:
            gt_label_class, gt_label_spots = [], []
            for ds in data_samples:
                if hasattr(ds, "pseudo_label"):
                    gt_label_class.append(getattr(ds.pseudo_label, 'gt_label_class', torch.tensor(-1)).item())
                    gt_label_spots.append(getattr(ds.pseudo_label, 'gt_label_spots', torch.tensor(-1)).item())
    
            gt_label_class = torch.as_tensor(gt_label_class, dtype=torch.long, device=classification.device)
            gt_label_spots = torch.as_tensor(gt_label_spots, dtype=torch.float, device=classification.device)

        assert len(classification) == len(gt_label_class)
        assert len(classification) == len(gt_label_spots)
                
        return dict(
            classification=classification.detach().cpu().numpy(),
            regression=regression.detach().cpu().numpy(),
            gt_label_spots=gt_label_spots.detach().cpu().numpy(),
            gt_label_class=gt_label_class.detach().cpu().numpy(),
            features=features[0].detach().cpu().numpy() if return_features else [None]*len(classification),
            images=inputs[0].detach().cpu().numpy() if return_images else [None]*len(classification),
            uncertainty=normalized_entropy.detach().cpu().numpy() if return_uncertainty else [None]*len(classification)
        )


    
    def val_step(self, data_batch, *args, **kwargs):
        
        device = next(self.backbone.parameters()).device
        
        inputs = data_batch['inputs']
        data_samples = data_batch['data_samples']
        
        assert isinstance(inputs, list)
        inputs = torch.stack(inputs, 1)
        inputs = inputs.reshape((inputs.size(0), inputs.size(2),inputs.size(3), inputs.size(4))).to(device)
        
        features = self.backbone(inputs)
        if self.reducer is not None:
            features = self.reducer(features)
            
        pred_logits = self.classifier(features)
        preds_class = torch.argmax(pred_logits, dim=1)
        
        for sample, pred_class in zip(data_samples, preds_class):
            setattr(sample, 'gt_label', LabelData())
            setattr(sample.gt_label, 'label', torch.tensor([sample.pseudo_label.gt_label_class]).to(device))
            setattr(sample, 'pred_label', LabelData())
            setattr(sample.pred_label, 'label', torch.tensor([pred_class]).to(device))
            
        return data_samples

from torch import nn

@MODELS.register_module()
class SimpleReducer(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        with_avg_pool: bool = True,
        backbone=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        self.with_avg_pool = with_avg_pool
        
        if with_avg_pool:
            self.reducer = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
                nn.Linear(in_channels, out_channels, bias=True),
            )
            
        else:
            # Spatial map preserved; channel reduction via 1x1 conv
            self.reducer = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def extract_feat(self, inputs: List[torch.Tensor], **kwargs) -> Tuple[torch.Tensor]:
        """Extract reduced features from the first input tensor."""
        features = self.reducer(inputs[0])
        return features 

    def forward(self, inputs: List[torch.Tensor], mode: str = "tensor", **kwargs):
        """Convenience forward that unwraps the single-tensor tuple for 'tensor' mode."""
        feats = self.extract_feat(inputs, **kwargs)
        return (feats,)
