import copy
from collections import deque
import numpy as np
from domainbed.lib import swa_utils


class SWADBase:
    def update_and_evaluate(self, segment_swa, val_acc, val_loss, prt_fn):
        raise NotImplementedError()

    def get_final_model(self):
        raise NotImplementedError()


class IIDMax(SWADBase):
    """SWAD start from iid max acc and select last by iid max swa acc"""

    def __init__(self, evaluator, **kwargs):
        self.iid_max_acc = 0.0
        self.swa_max_acc = 0.0
        self.avgmodel = None
        self.final_model = None
        self.evaluator = evaluator

    def update_and_evaluate(self, segment_swa, val_acc, val_loss, prt_fn):
        if self.iid_max_acc < val_acc:
            self.iid_max_acc = val_acc
            self.avgmodel = swa_utils.AveragedModel(segment_swa.module, rm_optimizer=True)
            self.avgmodel.start_step = segment_swa.start_step

        self.avgmodel.update_parameters(segment_swa.module)
        self.avgmodel.end_step = segment_swa.end_step

        # evaluate
        accuracies, summaries = self.evaluator.evaluate(self.avgmodel)
        results = {**summaries, **accuracies}
        prt_fn(results, self.avgmodel)

        swa_val_acc = results["train_out"]
        if swa_val_acc > self.swa_max_acc:
            self.swa_max_acc = swa_val_acc
            self.final_model = copy.deepcopy(self.avgmodel)

    def get_final_model(self):
        return self.final_model


class LossValley(SWADBase):
    """IIDMax has a potential problem that bias to validation dataset.
    LossValley choose SWAD range by detecting loss valley.
    """

    def __init__(self, evaluator, n_converge, n_tolerance, tolerance_ratio, **kwargs):
        """
        Args:
            evaluator
            n_converge: converge detector window size.
            n_tolerance: loss min smoothing window size
            tolerance_ratio: decision ratio for dead loss valley
        """
        self.evaluator = evaluator
        self.n_converge = n_converge
        self.n_tolerance = n_tolerance
        self.tolerance_ratio = tolerance_ratio

        self.converge_Q = deque(maxlen=n_converge)
        self.smooth_Q = deque(maxlen=n_tolerance)

        self.final_model = None

        self.converge_step = None
        self.dead_valley = False
        self.threshold = None

    def get_smooth_loss(self, idx):
        smooth_loss = min([model.end_loss for model in list(self.smooth_Q)[idx:]])
        return smooth_loss

    @property
    def is_converged(self):
        return self.converge_step is not None

    def update_and_evaluate(self, segment_swa, val_acc, val_loss, prt_fn):
        if self.dead_valley:
            return

        frozen = copy.deepcopy(segment_swa.cpu())
        frozen.end_loss = val_loss
        self.converge_Q.append(frozen)
        self.smooth_Q.append(frozen)

        if not self.is_converged:
            if len(self.converge_Q) < self.n_converge:
                return

            min_idx = np.argmin([model.end_loss for model in self.converge_Q])
            untilmin_segment_swa = self.converge_Q[min_idx]  # until-min segment swa.
            if min_idx == 0:
                self.converge_step = self.converge_Q[0].end_step
                self.final_model = swa_utils.AveragedModel(untilmin_segment_swa)

                th_base = np.mean([model.end_loss for model in self.converge_Q])
                self.threshold = th_base * (1.0 + self.tolerance_ratio)

                if self.n_tolerance < self.n_converge:
                    for i in range(self.n_converge - self.n_tolerance):
                        model = self.converge_Q[1 + i]
                        self.final_model.update_parameters(
                            model, start_step=model.start_step, end_step=model.end_step
                        )
                elif self.n_tolerance > self.n_converge:
                    converge_idx = self.n_tolerance - self.n_converge
                    Q = list(self.smooth_Q)[: converge_idx + 1]
                    start_idx = 0
                    for i in reversed(range(len(Q))):
                        model = Q[i]
                        if model.end_loss > self.threshold:
                            start_idx = i + 1
                            break
                    for model in Q[start_idx + 1 :]:
                        self.final_model.update_parameters(
                            model, start_step=model.start_step, end_step=model.end_step
                        )
                print(
                    f"Model converged at step {self.converge_step}, "
                    f"Start step = {self.final_model.start_step}; "
                    f"Threshold = {self.threshold:.6f}, "
                )
            return

        if self.smooth_Q[0].end_step < self.converge_step:
            return

        # converged -> loss valley
        min_vloss = self.get_smooth_loss(0)
        if min_vloss > self.threshold:
            self.dead_valley = True
            print(f"Valley is dead at step {self.final_model.end_step}")
            return

        model = self.smooth_Q[0]
        self.final_model.update_parameters(
            model, start_step=model.start_step, end_step=model.end_step
        )

    def get_final_model(self):
        if not self.is_converged:
            self.evaluator.logger.error(
                "Requested final model, but model is not yet converged; return last model instead"
            )
            return self.converge_Q[-1].cuda()

        if not self.dead_valley:
            self.smooth_Q.popleft()
            while self.smooth_Q:
                smooth_loss = self.get_smooth_loss(0)
                if smooth_loss > self.threshold:
                    break
                segment_swa = self.smooth_Q.popleft()
                self.final_model.update_parameters(segment_swa, step=segment_swa.end_step)

        return self.final_model.cuda()
