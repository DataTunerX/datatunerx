import os
import json
import time
from typing import TYPE_CHECKING
from datetime import timedelta

from transformers import TrainerCallback
from transformers.trainer_utils import has_length, PREFIX_CHECKPOINT_DIR

from prometheus.metrics import export_train_metrics, export_eval_metrics

if TYPE_CHECKING:
    from transformers import TrainingArguments, TrainerState, TrainerControl

LOG_FILE_NAME = "trainer_log.jsonl"


class LogCallback(TrainerCallback):

    def __init__(self, runner=None, metrics_export_address=None, uid=None):
        self.runner = runner
        self.in_training = False
        self.start_time = time.time()
        self.cur_steps = 0
        self.max_steps = 0
        self.elapsed_time = ""
        self.remaining_time = ""
        self.metrics_export_address = metrics_export_address
        self.uid = uid

    def timing(self):
        cur_time = time.time()
        elapsed_time = cur_time - self.start_time
        avg_time_per_step = elapsed_time / self.cur_steps if self.cur_steps != 0 else 0
        remaining_time = (self.max_steps - self.cur_steps) * avg_time_per_step
        self.elapsed_time = str(timedelta(seconds=int(elapsed_time)))
        self.remaining_time = str(timedelta(seconds=int(remaining_time)))

    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the beginning of training.
        """
        if state.is_local_process_zero:
            self.in_training = True
            self.start_time = time.time()
            self.max_steps = state.max_steps
            if os.path.exists(os.path.join(args.output_dir, LOG_FILE_NAME)) and args.overwrite_output_dir:
                print("Previous log file in this folder will be deleted.")
                os.remove(os.path.join(args.output_dir, LOG_FILE_NAME))

    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the end of training.
        """
        if state.is_local_process_zero:
            self.in_training = False
            self.cur_steps = 0
            self.max_steps = 0

    def on_substep_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the end of an substep during gradient accumulation.
        """
        if state.is_local_process_zero and self.runner is not None and self.runner.aborted:
            control.should_epoch_stop = True
            control.should_training_stop = True

    def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the end of a training step.
        """
        if state.is_local_process_zero:
            self.cur_steps = state.global_step
            self.timing()
            if self.runner is not None and self.runner.aborted:
                control.should_epoch_stop = True
                control.should_training_stop = True

    def on_evaluate(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called after an evaluation phase.
        """
        if state.is_local_process_zero and not self.in_training:
            self.cur_steps = 0
            self.max_steps = 0

    def on_predict(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", *other, **kwargs):
        r"""
        Event called after a successful prediction.
        """
        if state.is_local_process_zero and not self.in_training:
            self.cur_steps = 0
            self.max_steps = 0

    def on_log(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs) -> None:
        r"""
        Event called after logging the last logs.
        """
        if not state.is_local_process_zero:
            return

        print('log_history: ', state.log_history[-1])  # add 看看返回的 key
        if "eval_loss" in state.log_history[-1].keys():
            eval_log = dict(
                uid=self.uid,
                current_steps=self.cur_steps,
                total_steps=self.max_steps,
                eval_loss=state.log_history[-1].get("eval_loss", None),
                eval_perplexity=state.log_history[-1].get("eval_perplexity", None),
                eval_rouge_1=state.log_history[-1].get("eval_rouge-1", None),
                eval_rouge_2=state.log_history[-1].get("eval_rouge-2", None),
                eval_rouge_l=state.log_history[-1].get("eval_rouge-l", None),
                eval_bleu_4=state.log_history[-1].get("eval_bleu-4", None),
                epoch=state.log_history[-1].get("epoch", None),
                percentage=round(self.cur_steps / self.max_steps * 100, 2) if self.max_steps != 0 else 100,
                elapsed_time=self.elapsed_time,
                remaining_time=self.remaining_time
            )
        else:
            logs = dict(
                uid=self.uid,
                current_steps=self.cur_steps,
                total_steps=self.max_steps,
                loss=state.log_history[-1].get("loss", None),
                eval_loss=state.log_history[-1].get("eval_loss", None),
                val_perplexity=state.log_history[-1].get("eval_perplexity", None),
                eval_rouge_1=state.log_history[-1].get("eval_rouge-1", None),
                eval_rouge_2=state.log_history[-1].get("eval_rouge-2", None),
                eval_rouge_l=state.log_history[-1].get("eval_rouge-l", None),
                eval_bleu_4=state.log_history[-1].get("eval_bleu-4", None),
                predict_loss=state.log_history[-1].get("predict_loss", None),
                reward=state.log_history[-1].get("reward", None),
                learning_rate=state.log_history[-1].get("learning_rate", None),
                epoch=state.log_history[-1].get("epoch", None),
                percentage=round(self.cur_steps / self.max_steps * 100, 2) if self.max_steps != 0 else 100,
                elapsed_time=self.elapsed_time,
                remaining_time=self.remaining_time
            )
        if self.runner is not None:
            print("{{'loss': {:.4f}, 'learning_rate': {:2.4e}, 'epoch': {:.2f}}}".format(
                logs["loss"] or 0, logs["learning_rate"] or 0, logs["epoch"] or 0
            ))

        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'watch'), exist_ok=True)
        if "eval_loss" in state.log_history[-1].keys():
            with open(os.path.join(args.output_dir, 'watch', "eval_log.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(eval_log) + "\n")
            if self.metrics_export_address:
                export_eval_metrics(self.metrics_export_address, eval_log)
        else:
            with open(os.path.join(args.output_dir, 'watch', "trainer_log.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(logs) + "\n")
            if self.metrics_export_address:
                export_train_metrics(self.metrics_export_address, logs)

    def on_prediction_step(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called after a prediction step.
        """
        eval_dataloader = kwargs.pop("eval_dataloader", None)
        if state.is_local_process_zero and has_length(eval_dataloader) and not self.in_training:
            if self.max_steps == 0:
                self.max_steps = len(eval_dataloader)
            self.cur_steps += 1
            self.timing()
