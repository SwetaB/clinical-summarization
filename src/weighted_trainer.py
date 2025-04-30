import torch
from transformers import Trainer


class WeightedChunkTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        chunk_ids = inputs.pop("chunk_id")

        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss = loss.view(labels.size())

        weights = 1.0 / (chunk_ids.float() + 1.0)
        weights = weights.unsqueeze(1).expand_as(loss)

        weighted_loss = (loss * weights).mean()

        return (weighted_loss, outputs) if return_outputs else weighted_loss


class TrainerWithTrainLoss(Trainer):
    def _log(self, logs: dict) -> None:
        if "loss" in logs and self.control.should_log:
            logs["train_loss"] = logs["loss"]
        super()._log(logs)