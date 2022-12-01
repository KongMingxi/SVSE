import numpy as np
import torch
from torchvision.utils import make_grid
from base import TrainerBase
from utils import inf_loop, MetricTracker

from logger import train_logger


class BaselineTrainer(TrainerBase):
    """Baseline model.
    
    A.k.a. Empirical Risk Minimization, or ERM.
    """
    def __init__(self, cfg):
        super().__init__()

        if torch.cuda.is_available() and cfg.use_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Save as attributes some frequently used variables        
        self.cfg = cfg
        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.optim.max_epoch
        self.output_dir = cfg.output_dir
        self.best_result = -np.inf

        self.build_data_loader()
        self.build_model()
        self.evaluator = build_evaluator(cfg, lab2cname=self.lab2cname)

    def train(self):
        super().train(self.start_epoch, self.max_epoch)

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)


    def forward_backward(self, batch):
        input, target = self.parse_batch_train(batch)
        output = self.model(input)
        loss = F.cross_entropy(output, target)
        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, target)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        target = batch["label"]
        input = input.to(self.device)
        target = target.to(self.device)
        return input, target
