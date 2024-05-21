from copy import deepcopy
from typing import Callable, Optional
from addict import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from fcore.utils.types import TDataList
from fcore.utils.misc import sharpen, entropy
from fcore.utils.checkpoint import Checkpointer
from fcore.utils.threshold import ThreshController
from fcore.utils.logger import logger
from fcore.model import GenericModule
from fcore.utils.ohem import ProbOhemCrossEntropy2d

from seg.model.segmentor.ts import TSSegmentor
from seg.model.loss.euup import EUUPLoss
from seg.model.loss.functional import DiceLoss, make_one_hot

class WeightGenerator(GenericModule):
    def __init__(self, feats_channels=256) -> None:
        super().__init__()
        
        self.feats_channels = feats_channels

        self.transformer_encoder = nn.TransformerEncoderLayer(
            d_model=feats_channels, nhead=1, dim_feedforward=64, batch_first=True
        )
    
    def forward_train(
        self,
        init_feats: torch.Tensor, 
        init_weights: torch.Tensor, 
        labels: torch.Tensor = None, 
        scores: torch.Tensor = None, 
    ) -> torch.Tensor:      
        if labels is not None:
            labels = (
                F.interpolate(labels.unsqueeze(1).float(), size=init_feats.shape[2:], mode="nearest").squeeze(1).long()
            ) 
            scores = F.interpolate(
                scores.unsqueeze(1), size=init_feats.shape[2:], mode="bilinear", align_corners=False
            ).squeeze(1)

            sampled_mask = torch.zeros_like(labels, dtype=torch.bool)

            unique_classes, unique_n = labels.unique(return_counts=True)

            for cls, n in zip(unique_classes, unique_n):
                if cls.item() == 255:
                    continue
                cls_scores = scores[labels == cls]
                cls_scores = cls_scores.sort(descending=True)[0]
                cls_threshold = cls_scores[int(n * 0.2)]
                sampled_mask = torch.where(labels == cls, scores > cls_threshold, sampled_mask)
            feats = init_feats.permute(0, 2, 3, 1).reshape(-1, 1, self.feats_channels)[sampled_mask.view(-1)]
            init_weights = init_weights.view(1, -1, self.feats_channels)
            x = torch.cat((feats, init_weights.repeat(feats.shape[0], 1, 1)), dim=1)
            out = self.transformer_encoder(x)
            out = out[:, 0] 
            return out, sampled_mask
        else:
            return self.forward_eval(init_feats, init_weights)
        
    def forward_eval(self, init_feats: torch.Tensor, init_weights: torch.Tensor) -> torch.Tensor:
        feats = init_feats.permute(0, 2, 3, 1).reshape(-1, 1, self.feats_channels)
        init_weights = init_weights.view(1, -1, self.feats_channels)

        x = torch.cat((feats, init_weights.repeat(feats.shape[0], 1, 1)), dim=1)

        out = self.transformer_encoder(x)

        out = out[:, 0].view(init_feats.shape[0], init_feats.shape[2], init_feats.shape[3], -1).permute(0, 3, 1, 2)
        return out
    
class EUUPSegmentor(TSSegmentor):
    def __init__(self, cfg, segmentor_class: Callable[..., GenericModule]):
        super(EUUPSegmentor, self).__init__(cfg, segmentor_class)

        self._euup_loss = EUUPLoss(type="focal_loss")

        self.feats_s: Optional[torch.Tensor] = None
        self.feats_t: Optional[torch.Tensor] = None

        self.segmentor.classifier.register_forward_hook(self._hook_student_feats)
        self.ema_segmentor.classifier.register_forward_hook(self._hook_teacher_feats)

        self.weight_generator = WeightGenerator(feats_channels=256)
        self.total_step = cfg.solver.iter_num
        self.unreliable_ratio = cfg.solver.unreliable_ratio
        self.thresh_init = cfg.solver.thresh_init
        self.checkpointer = Checkpointer(cfg)
        self.thresh_controller = ThreshController(nclass=21, momentum=0.999, thresh_init=self.thresh_init)
        if self.cfg.resume is None:
            self.cur_step = 0
        else:
            self.cur_step = self.checkpointer.load(self.cfg.resume, model=self.cfg.model)

    def _hook_student_feats(self, hooked_module, input, output):
        self.feats_s = input[0]

    def _hook_teacher_feats(self, hooked_module, input, output):
        self.feats_t = input[0].detach() 
    
    @torch.no_grad()
    def _pseudo_labeling(self, data_list: TDataList, sharpen_factor: float = 0.5) -> torch.Tensor:
        self.switch_to_bn_statistics_pl() 
        self.ema_segmentor.train()
        self.forward_generic(data_list, selector="teacher")
        self.ema_segmentor.eval()

        logits = self.forward_generic(data_list, selector="teacher")
        pseudo_probs = F.softmax(logits, dim=1) 
        if sharpen_factor > 0:
            pseudo_probs = sharpen(pseudo_probs, sharpen_factor)

        self.switch_to_bn_statistics_eval()

        self.segmentor.eval()         
        logits = self.forward_generic(data_list, selector = "student")
        potential_probs_extra = F.softmax(logits, dim=1)
        if sharpen_factor > 0:
            potential_probs_extra = sharpen(potential_probs_extra, sharpen_factor)
        self.segmentor.train()

        self.segmentor.eval()
        data_list_flipped = deepcopy(data_list)
        for data in data_list_flipped:
            data.img = torch.flip(data.img, dims=(2,))

        logits = self.forward_generic(data_list_flipped, selector="student")
        logits = torch.flip(logits, dims=(3,))
        potential_probs = F.softmax(logits, dim=1)
        if sharpen_factor > 0:
            potential_probs = sharpen(potential_probs, sharpen_factor)

        self.segmentor.train()

        return pseudo_probs.detach(), potential_probs.detach(), potential_probs_extra.detach()

    def forward_train(
        self,
        data_list_l_weak: TDataList,
        data_list_u_weak: TDataList,
        data_list_l_strong: TDataList,
        data_list_u_strong: TDataList,
    ):
        logits_l = self.forward_generic(data_list_l_strong, selector="student")
        labels_l = torch.cat([data.label for data in data_list_l_strong], dim=0).long()
        euup_weights_l, euup_weights_l_sampled_mask = self.weight_generator(
            self.feats_s.detach(), self.segmentor.classifier.weight.detach(), labels_l, torch.max(logits_l, dim=1)[0]
        )
        with torch.no_grad():
            euup_weights_l_target = (
                self.segmentor.classifier.weight.detach()[
                    F.interpolate(labels_l.unsqueeze(1).float(), size=self.feats_s.shape[2:4], mode="nearest")
                    .long()
                    .clamp_(max=logits_l.shape[1] - 1) 
                    .view(-1)
                ]
                .squeeze(2)
                .squeeze(2)
            )
            euup_weights_l_target = euup_weights_l_target[euup_weights_l_sampled_mask.view(-1)]
        
        loss_dict_l = self.get_losses(
            logits=logits_l,
            labels=labels_l,
            gt_labels=labels_l,
            euup_weights=euup_weights_l,
            euup_weights_target=euup_weights_l_target,
            euup_weights_gt_target=euup_weights_l_target,
            labelled=True,
        )

        logits_u = self.forward_generic(data_list_u_strong, selector="student")
        
        euup_weights_u = self.weight_generator(self.feats_s.detach(), self.segmentor.classifier.weight.detach())
        with torch.no_grad():
            if self.cfg.solver.euup_weight_norm == "adaptive":
                w_norm = self.segmentor.classifier.weight.norm(dim=1).mean().detach()
            if self.cfg.solver.euup_weight_norm == "adaptive_min":
                w_norm = self.segmentor.classifier.weight.norm(dim=1).min().detach()
            else:
                w_norm = 1 / self.cfg.solver.euup_weight_norm

        euup_weights_u = F.normalize(euup_weights_u, dim=1) * w_norm
        euup_logits_u = torch.einsum("bchw,bchw->bhw", self.feats_s, euup_weights_u.detach()).unsqueeze(1)
        euup_logits_u = F.interpolate(euup_logits_u, size=logits_u.shape[2:], mode="bilinear", align_corners=False)
        logits_u = torch.cat((logits_u, euup_logits_u), dim=1)
        
        pseudo_labels_u, potential_labels_u, potential_labels_u_extra = self._pseudo_labeling(data_list_u_weak)

        labels_u = torch.cat([data.label for data in data_list_u_weak], dim=0).long()
        labels_u_strong = torch.cat([data.label for data in data_list_u_strong], dim=0).long()

        with torch.no_grad():
            euup_weights_u_target = (
                self.segmentor.classifier.weight.detach()[
                    F.interpolate(
                        torch.argmax(pseudo_labels_u, dim=1).unsqueeze(1).float(),
                        size=self.feats_s.shape[2:4], 
                        mode="nearest",
                    )
                    .long()
                    .clamp_(max=self.cfg.model.num_classes - 1)
                    .squeeze(1)
                ]
                .squeeze(4)
                .squeeze(4)
                .permute(0, 3, 1, 2)
            )

            euup_weights_u_gt_target = (
                self.segmentor.classifier.weight.detach()[
                    F.interpolate(labels_u_strong.unsqueeze(1).float(), size=self.feats_s.shape[2:4], mode="nearest")
                    .long()
                    .clamp_(max=self.cfg.model.num_classes - 1)
                    .squeeze(1)
                ]
                .squeeze(4)
                .squeeze(4)
                .permute(0, 3, 1, 2)
            )
        logits_u_weak = self.forward_generic(data_list_u_weak, selector="teacher")
        loss_dict_u, euup_stat_dict = self.get_losses(
            logits=logits_u,
            logits_u_weak = logits_u_weak,
            labels=pseudo_labels_u,
            gt_labels=labels_u,
            potential_labels=potential_labels_u,
            potential_labels_extra=potential_labels_u_extra,
            euup_weights=euup_weights_u,
            euup_weights_target=euup_weights_u_target,
            euup_weights_gt_target=euup_weights_u_gt_target,
            labelled=False,
        )

        loss_dict = Dict()
        loss_dict.loss_l = loss_dict_l
        loss_dict.loss_u = loss_dict_u

        stat_dict = self._pseudo_labeling_statistics(pseudo_labels_u, labels_u)
        stat_dict.update(euup_stat_dict)

        return loss_dict, stat_dict

    def get_losses(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        logits_u_weak = None,
        gt_labels: Optional[torch.Tensor] = None,
        potential_labels: Optional[torch.Tensor] = None,
        potential_labels_extra: Optional[torch.Tensor] = None,
        euup_weights: Optional[torch.Tensor] = None,
        euup_weights_target: Optional[torch.Tensor] = None,
        euup_weights_gt_target: Optional[torch.Tensor] = None,
        labelled: bool = True,
    ) -> Dict:
        loss_dict = Dict()
        if labelled:
            
            with torch.no_grad():
                entropy_scores = entropy(logits.detach())
                entropy_mask_list = []
                for i in range(len(entropy_scores)):
                    if len(torch.sort(entropy_scores[i][gt_labels[i] != 255].detach().flatten(), descending=False)[0]) == 0:
                        entropy_mask_list.append(torch.zeros_like(entropy_scores[i], dtype=bool))
                    else:
                        entropy_sort = torch.sort(entropy_scores[i][gt_labels[i] != 255].detach().flatten(), descending=False)[0]
                        index = round(len(entropy_sort) * self.unreliable_ratio  - 1)
                        entropy_threshold = entropy_sort[index]
                        entropy_mask_ = torch.ge(entropy_scores[i], entropy_threshold)
                        entropy_mask_list.append(entropy_mask_)
                    entropy_mask = torch.stack(entropy_mask_list, dim=0)
            gt_l = torch.where(entropy_mask, gt_labels, 255)
            logits_l = self.segmentor.classifier(self.feats_s)

            logits_l = F.interpolate(logits_l, (512, 512), mode="bilinear", align_corners=True)
            loss_edge = F.cross_entropy(logits_l, gt_l.long(), ignore_index=255)
            
            entropy_mask_weight = torch.ones_like(entropy_mask).float() 
            entropy_mask_weight = entropy_mask_weight + entropy_scores

            loss = F.cross_entropy(logits, labels.long(), ignore_index=255)
            loss = loss * torch.ne(gt_labels, 255).float() * entropy_mask_weight
            loss = loss.sum() / (torch.ne(gt_labels, 255).sum() + 1e-10)

            loss_vw = -F.cosine_similarity(euup_weights, euup_weights_target).mean()
            loss_dict.loss = loss
            loss_dict.loss_vw = loss_vw
            loss_dict.loss_bou = loss_edge

            return loss_dict
        else:
            pseudo_probs_u, pseudo_preds_u = torch.max(labels, dim=1) 
                    
            valid_mask = torch.ne(gt_labels, 255)
            
            self.thresh_controller.thresh_update(labels.detach(), gt=gt_labels.detach(), update_g=True)
            score_threshold = self.thresh_controller.get_thresh_global()
            low_confidence_mask = torch.lt(pseudo_probs_u, score_threshold)
            _, p_labels1 = torch.max(labels, dim=1)
            _, p_labels2 = torch.max(potential_labels, dim=1)
            p_labels2 = torch.where(low_confidence_mask, torch.sort(labels, dim=1, descending=True)[1][:, 1], p_labels2)
            _, p_labels3 = torch.max(potential_labels_extra, dim=1)
            p_labels3 = torch.where(low_confidence_mask, torch.sort(labels, dim=1, descending=True)[1][:, 2], p_labels3)

            with torch.no_grad():
                entropy_scores = entropy(logits_u_weak.detach())
                entropy_mask_list = []
                for i in range(len(entropy_scores)):
                    if len(torch.sort(entropy_scores[i][gt_labels[i] != 255].detach().flatten(), descending=False)[
                                0]) == 0:
                        entropy_mask_list.append(torch.zeros_like(entropy_scores[i], dtype=bool))
                    else:
                        entropy_sort = \
                        torch.sort(entropy_scores[i][gt_labels[i] != 255].detach().flatten(), descending=False)[0]
                        index = round(len(entropy_sort) * self.unreliable_ratio - 1)
                        entropy_threshold = entropy_sort[index]
                        entropy_mask_ = torch.ge(entropy_scores[i], entropy_threshold)
                        entropy_mask_list.append(entropy_mask_)
                    entropy_mask = torch.stack(entropy_mask_list, dim=0)

            with torch.no_grad():
                logits_u_weak_target = self.segmentor.classifier(self.feats_t.detach())
                logits_u_weak_target = F.interpolate(logits_u_weak_target, (512, 512), mode="bilinear",align_corners=True )
                p_target = torch.max(logits_u_weak_target, dim=1)[1]
                logits_u_weak_target = torch.where(entropy_mask & valid_mask, p_target, 255)
                p_target = torch.where(entropy_mask, p_target, p_labels1)


            p_labels = torch.stack((p_labels1, p_labels2, p_labels3, p_target), dim=1)
            euup_mask = ~torch.all(torch.eq(p_labels, p_labels[:, 0].unsqueeze(1)), dim=1)

            euup_mask_weight = torch.ones_like(euup_mask).float() 

            euup_mask_weight = euup_mask_weight + entropy_scores

            loss = self._euup_loss(logits, p_labels1, p_labels2, p_labels3, p_target, reduction="none")

            loss = loss * valid_mask.float() * euup_mask_weight
            loss = loss.sum() / (valid_mask.sum() + 1e-10)

            euup_ratio = (euup_mask.float() * valid_mask.float()).sum() / (valid_mask.float().sum() + 1e-10)
            with torch.no_grad():
                vcw_mask = F.interpolate(
                    (valid_mask & ~euup_mask).unsqueeze(1).float(), size=euup_weights.shape[2:4], mode="nearest",
                ).squeeze(1)

            loss_vw = (-F.cosine_similarity(euup_weights, euup_weights_target) * vcw_mask.float()).sum() / (
                vcw_mask.float().sum() + 1e-10
            )

            with torch.no_grad():
                vcw_mask = F.interpolate(
                    (valid_mask & euup_mask).unsqueeze(1).float(), size=euup_weights.shape[2:4], mode="nearest",
                ).squeeze(1)
                sim_matrix = torch.cosine_similarity(euup_weights, euup_weights_gt_target, dim=1)
                euup_weights_sim = (sim_matrix * vcw_mask.float()).sum() / (vcw_mask.float().sum() + 1e-10)

            loss_dict.loss = loss 
            loss_dict.loss_vw = loss_vw                

            return loss_dict, Dict(euup_weights_sim=euup_weights_sim, euup_ratio=euup_ratio)