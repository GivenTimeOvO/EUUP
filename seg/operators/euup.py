import torch.nn as nn
import torch
from torch.nn.parallel import DistributedDataParallel
from addict import Dict

from fcore.optim.utils import sum_loss
import fcore.utils.distributed as dist_utils
from fcore.operators import OPERATOR_REGISTRY

from fcore.optim import make_lr_scheduler, make_optimizer
from seg.model.segmentor.deeplabv3p_alt import DeeplabV3pAlt
from seg.model.segmentor.euup import EUUPSegmentor
from seg.operators.ts import TeacherStudentOpt

@OPERATOR_REGISTRY.register("euup")
class VirtualCategoryOpt(TeacherStudentOpt):
    def __init__(self, cfg):
        super(VirtualCategoryOpt, self).__init__(cfg)
        
        self.generator_optimizer = None
        self.scaler = torch.cuda.amp.GradScaler()
        self.amp_train = cfg.solver.amp_train
    
    def build_model(self):
        model = EUUPSegmentor(self.cfg, DeeplabV3pAlt)
        model.to(self.device)
        if dist_utils.is_distributed():
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DistributedDataParallel(model, find_unused_parameters=True)

        self.model = model
        
    def build_optimizer(self):
        self.optimizer = torch.optim.SGD([{'params': self.model.module.segmentor.backbone.parameters() if dist_utils.is_distributed() else self.model.segmentor.backbone.parameters(), 
                                           'lr': self.cfg.solver.optimizer.lr},
                    {'params': [param for name, param in (self.model.module.segmentor.named_parameters() if dist_utils.is_distributed() else self.model.segmentor.named_parameters()) if 'backbone' not in name],
                    'lr': self.cfg.solver.optimizer.lr * self.cfg.solver.lr_multi}], lr=self.cfg.solver.optimizer.lr, momentum=0.9, weight_decay=1e-4)


        self.generator_optimizer = make_optimizer(
            self.cfg.solver.optimizer,
            self.model.module.weight_generator if dist_utils.is_distributed() else self.model.weight_generator,
        )

        self.lr_scheduler = make_lr_scheduler(self.cfg.solver.lr_scheduler, self.optimizer)

    def train_pre_step(self) -> Dict:
        self.generator_optimizer.zero_grad()
        return super().train_pre_step()
    
    def train_run_step(self, model_inputs: Dict, **kwargs) -> Dict:
        while self.amp_train:
            with torch.cuda.amp.autocast():
                loss_dict, stat_dict = self.model(**model_inputs)
                losses = sum_loss(loss_dict.loss_l) + (
                    self.cfg.solver.u_loss_weight if self.cur_step > self.burnin_step else 0
                ) * sum_loss(loss_dict.loss_u)
            self.scaler.scale(losses).backward()

            self.scaler.unscale_(self.optimizer)
            self.scaler.unscale_(self.generator_optimizer)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.scaler.step(self.optimizer)
            self.scaler.step(self.generator_optimizer)
            self.scaler.update()

            self.lr_scheduler.step()
            if dist_utils.is_distributed():
                self.model.module._momentum_update(self.cfg.solver.ema_momentum)
            else:
                self.model._momentum_update(self.cfg.solver.ema_momentum)

            return Dict(loss_dict=loss_dict, stat_dict=stat_dict)

        else:
            loss_dict, stat_dict = self.model(**model_inputs)
            
            losses = sum_loss(loss_dict.loss_l) + (
                self.cfg.solver.u_loss_weight if self.cur_step > self.burnin_step else 0
            ) * sum_loss(loss_dict.loss_u)
            
            losses.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            self.optimizer.step()
            self.generator_optimizer.step()
            self.lr_scheduler.step()
            
            if dist_utils.is_distributed():
                self.model.module._momentum_update(self.cfg.solver.ema_momentum)
            else:
                self.model._momentum_update(self.cfg.solver.ema_momentum)
            
            return Dict(loss_dict=loss_dict, stat_dict=stat_dict)

