from typing import Optional, Tuple
import torch
import torch.nn.functional as F

from fcore.utils.misc import entropy

class EUUPLoss:
    def __init__(
            self,
            confidence_threshold: float = 0.0,
            type: str = "mse", # mean squared error (MSE) loss function
            num_classes: Optional[int] = None,
            gamma: float = 1.0,
    ):
        if type == "mse" or type == "mean_squared_error":
            self.loss_fn = self._mse
        elif type == "ce" or type == "cross_entropy":
            self.loss_fn = self._cross_entropy
        elif type == "soft_ce" or type == "soft_cross_entropy":
            self.loss_fn = self._soft_cross_entropy
        elif type == "focal_loss":
            self.loss_fn = self._focal_loss
            self.gamma = gamma
        elif type == "mse_full_euup":
            self.loss_fn = self._mse_full_euup
        else:
            raise ValueError("Type {} is not supported.".format(type))

        self.confidence_threshold = confidence_threshold
        self.num_classes = num_classes
        
    def _avoid_nan_backward_hook(self, logits: torch.Tensor):
        logits.register_hook(lambda grad: torch.nan_to_num(grad, 0, 0, 0)) 

    def _masked_softmax(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # masked_softmax: softmax with ignoring index argument
        mask = torch.ge(targets, 0)  

        logits = logits - logits.max(dim=1, keepdim=True)[0] 
        logits_exp = torch.exp(logits)
        probs = logits_exp / ((logits_exp * mask.float()).sum(dim=1, keepdim=True) + 1e-10) 
        return probs, mask

    def _mse(self, logits: torch.Tensor, targets: torch.Tensor, redunction="none") -> torch.Tensor:
        probs, mask = self._masked_softmax(logits, targets)

        mse = F.mse_loss(probs, targets, reduction="none") * mask.float()

        if redunction == "mean":
            mse = (mse.sum(dim=1) / mask.sum(dim=1).float()).mean()
        elif redunction == "sum":
            mse = mse.sum(dim=1)
        elif redunction == "none":
            pass

        return mse

    def _mse_full_euup(self, logits: torch.Tensor, targets: torch.Tensor, redunction="none") -> torch.Tensor:
        non_euup_flag = (targets[:, -1] == -1).unsqueeze(1)

        targets_max, targets_max_idx = targets.max(dim=1, keepdim=True) 
        euup_pos_idx = torch.cat((targets_max_idx, targets_max_idx), dim=1)
        euup_pos_idx[:, -1] = targets.shape[1] - 1

        euup_pos_idx[~non_euup_flag.expand_as(euup_pos_idx)] = targets.shape[1] - 1
        targets_max[non_euup_flag.expand_as(targets_max)] = targets_max[non_euup_flag.expand_as(targets_max)] / 2.0

        targets = targets.scatter(1, euup_pos_idx, targets_max.expand_as(euup_pos_idx))

        probs, mask = self._masked_softmax(logits, targets)

        mse = F.mse_loss(probs, targets, reduction="none") * mask.float()

        if redunction == "mean":
            mse = (mse.sum(dim=1) / mask.sum(dim=1).float()).mean()
        elif redunction == "sum":
            mse = mse.sum(dim=1)
        elif redunction == "none":
            pass

        return mse

    def _focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape[0] == 0:
            return 0 * logits.sum()

        mask = torch.ge(targets, 0).float()

        logits_exp = logits.exp()

        pos_term = (1 / logits_exp * targets * mask).sum(dim=1)
        neg_term = (logits_exp * torch.eq(targets, 0).float() * mask).sum(dim=1)

        ce = (1 + pos_term * neg_term).log()

        p = torch.exp(-ce)
        loss = (1 - p) ** self.gamma * ce

        return loss

    def _cross_entropy(self, logits: torch.Tensor, targets: torch.Tensor, redunction="none") -> torch.Tensor:
        mask = torch.ge(targets, 0).float() 

        logits_exp = logits.exp()

        pos_term = (1 / logits_exp * targets * mask).sum(dim=1)  
        neg_term = (logits_exp * torch.eq(targets, 0).float() * mask).sum(dim=1) 

        ce = (1 + pos_term * neg_term).log()

        if redunction == "mean":
            ce = (ce.sum(dim=1) / mask.sum(dim=1).float()).mean()
        elif redunction == "sum":
            ce = ce.sum(dim=1)
        elif redunction == "none":
            pass

        return ce

    def _soft_cross_entropy(self, logits: torch.Tensor, targets: torch.Tensor, redunction="none") -> torch.Tensor:
        mask = torch.ge(targets, 0).float()

        logits_exp = (logits - (logits.max(dim=1, keepdim=True)[0] - logits.min(dim=1, keepdim=True)[0]) / 2.0).exp()
        probs = logits_exp / (logits_exp * mask.float()).sum(dim=1, keepdim=True)

        probs_log = probs.log()

        ce = (-targets * probs_log) * mask.float()

        if redunction == "mean":
            ce = (ce.sum(dim=1) / mask.sum(dim=1).float()).mean()
        elif redunction == "sum":
            ce = ce.sum(dim=1)
        elif redunction == "none":
            pass

        return ce

    def _get_target_clses_and_probs(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if logits.dim() - 1 == targets.dim():
            clses = targets
            # 2*512*512
            num_classes = self.num_classes if self.num_classes is not None else logits.shape[1] - 1
            probs = torch.zeros((logits.shape[0], num_classes, *logits.shape[2:]), device=logits.device).scatter_(
                1, targets.unsqueeze(1), 1
            )
 
        else:
            clses = targets.argmax(dim=1)  
            probs = targets

        return clses, probs

    def __call__(
        self, logits: torch.Tensor, targets: torch.Tensor, potential_targets: torch.Tensor, potential_targets1: torch.Tensor, boundary_target: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        assert logits.dim() == targets.dim() or logits.dim() - 1 == targets.dim(), "logits or targets shape is invalid."
        assert (
            logits.dim() == potential_targets.dim() or logits.dim() - 1 == potential_targets.dim()
        ), "logits or potential_targets shape is invalid."
        self._avoid_nan_backward_hook(logits)

        with torch.no_grad():
            t_clses, t_probs = self._get_target_clses_and_probs(logits, targets)
            pt_clses, pt_probs = self._get_target_clses_and_probs(logits, potential_targets)
            pt1_clses, pt1_probs = self._get_target_clses_and_probs(logits, potential_targets1)
            b_clses, b_probs = self._get_target_clses_and_probs(logits, boundary_target)
            euup_target = t_probs.clone()
            ignore_clses = torch.stack((t_clses, pt_clses, pt1_clses, b_clses), dim=1)

            euup_prob = torch.gather(t_probs, 1, ignore_clses).sum(dim=1, keepdim=True)
            euup_prob = torch.where(euup_prob==3, 1, euup_prob)

            euup_target = torch.cat((euup_target, euup_prob), dim=1)

            no_euup_mask = torch.all(torch.eq(ignore_clses, ignore_clses[:, 0].unsqueeze(1)), dim=1)
            ignore_clses.masked_fill_(no_euup_mask.unsqueeze(1), logits.shape[1] - 1)
            euup_target = euup_target.scatter(1, ignore_clses, -1)

        loss = self.loss_fn(logits, euup_target)

        if self.confidence_threshold > 0.0:
            confidence_mask = torch.ge(t_probs.max(dim=1)[0], self.confidence_threshold)
            loss = loss * confidence_mask.unsqueeze(1).float()

        loss = torch.nan_to_num(loss, 0, 0, 0)

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum(dim=1)
        elif reduction == "none":
            pass

        return loss