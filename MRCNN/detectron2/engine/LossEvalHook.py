from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
import torch
import time
import datetime
import logging
import numpy as np


class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, cfg, data_loader):
        self._model = model
        self._period = eval_period
        self._cfg = cfg
        self._data_loader = data_loader
        self._last_loss = 100
        self._trigger_times = 0
        self._checkpointer = DetectionCheckpointer(
                # Assume you want to save checkpoints together with logs/statistics
                self._model,
                self._cfg.OUTPUT_DIR,
            )
        
    def limitcomma(self, value, limit=2):
        v = str(value).split(".")
        if len(v)==1:
            v.append('0')
        v[1]=v[1]+'0'*limit
        return float(v[0]+"."+v[1][:limit])

    
    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
            
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):            
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()
        
        #early stopping
        if self.limitcomma(self._last_loss,3) <= self.limitcomma(mean_loss,3):
            self._trigger_times += 1
            
            if self._trigger_times >= 10:
                self.trainer.storage.put_scalar('early_stop', self.trainer.iter)
                additional_state = {"iteration": self.trainer.iter}
                self._checkpointer.save("early_stoping_model")
                self.trainer.after_train()
                raise StopIteration
                
        else:
            self._last_loss = mean_loss
            self._trigger_times=0

        return losses
            
    
    def _get_loss(self, data):
        # How loss is calculated on train_loop 
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced
        
        
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)
        
        
