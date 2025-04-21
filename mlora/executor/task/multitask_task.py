import logging
from collections import OrderedDict
from typing import List, Tuple, override
import torch
import torch.nn.functional as F

import mlora.profiler
from mlora.executor.task.train_task import TrainTask
from mlora.model.args import LinearInfo, MLoRADataConfig, Tokens
from mlora.model.tokenizer import Tokenizer
from mlora.executor.context import TrainLoRAContext
from mlora.config import MultitaskConfig

class MultitaskTask(TrainTask):
    context_: TrainLoRAContext
    config_: MultitaskConfig

    @override
    def prepare(self, linears_info: OrderedDict[str, LinearInfo], tokenizer: Tokenizer):
        self.tokenizer_ = tokenizer
        # Prepare dataset and context
        self._pre_dataset()
        self._pre_context(linears_info)
        
        # Set custom loss function
        self.context_.set_loss_fn(self._multitask_loss)

    def _multitask_loss(self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Custom multitask loss that combines:
        1. Language modeling loss
        2. Classification loss
        3. Contrastive loss
        """
        # Unpack input based on task type
        batch_size = input.size(0)
        vocab_size = input.size(-1)
        
        # 1. Language Modeling Loss
        lm_loss = F.cross_entropy(
            input.view(-1, vocab_size),
            target.view(-1),
            ignore_index=-100
        )

        # 2. Classification Loss (assuming last token represents class)
        cls_logits = input[:, -1, :]  # Take last token's predictions
        cls_labels = target[:, -1]    # Take last token as class label
        classification_loss = F.cross_entropy(cls_logits, cls_labels)

        # 3. Contrastive Loss
        # Create embeddings from the last hidden state
        embeddings = input[:, 0, :]  # Using first token ([CLS]) representation
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T)
        
        # Create positive/negative masks for contrastive learning
        labels = target[:, 0]  # Using first token for creating pairs
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        mask = mask - torch.eye(batch_size, device=mask.device)  # Remove self-similarity
        
        # Temperature-scaled contrastive loss
        temperature = self.config_.temperature_
        contrastive_loss = -torch.log(
            torch.exp(similarity_matrix / temperature) /
            (torch.exp(similarity_matrix / temperature).sum(dim=1, keepdim=True) + 1e-7)
        ).mean()

        # Combine losses with weights from config
        total_loss = (
            self.config_.lm_weight_ * lm_loss +
            self.config_.cls_weight_ * classification_loss +
            self.config_.contrastive_weight_ * contrastive_loss
        )

        # Log individual losses
        mlora.profiler.metric_log(
            self.context_.path_ + "_lm_loss", lm_loss.item(), self.now_step_
        )
        mlora.profiler.metric_log(
            self.context_.path_ + "_cls_loss", classification_loss.item(), self.now_step_
        )
        mlora.profiler.metric_log(
            self.context_.path_ + "_contrastive_loss", contrastive_loss.item(), self.now_step_
        )
        mlora.profiler.metric_log(
            self.context_.path_ + "_total_loss", total_loss.item(), self.now_step_
        )

        logging.info(
            f"Task {self.context_.name_} - "
            f"LM Loss: {lm_loss.item():.4f}, "
            f"CLS Loss: {classification_loss.item():.4f}, "
            f"Contrastive Loss: {contrastive_loss.item():.4f}, "
            f"Total Loss: {total_loss.item():.4f}"
        )

        return total_loss

    @override
    def data(self, start_idx: int) -> Tuple[List[Tokens], List[MLoRADataConfig]]:
        logging.info(
            f"Adapter {self.context_.name_} "
            f"epoch: {self.now_epoch_}/{self.config_.num_epochs_} "
            f"iteration: {self.now_data_idx_}/{len(self.data_)} "
            f"step: {self.now_step_}"
        )
        
        data_idx_s = self.now_data_idx_
        data_idx_e = self.now_data_idx_ + self.config_.mini_batch_size_

        # Get the training data batch
        batch_data = self.data_[data_idx_s:data_idx_e]
        
        # Process each example in the batch
        ret_tokens = []
        for data_point in batch_data:
            # Combine text and metadata for multitask learning
            text = data_point["text"]
            label = data_point.get("label", "0")  # Default to "0" if no label
            
            # Create combined input with special tokens
            combined_input = f"[CLS] {text} [SEP] {label} [EOS]"
            
            # Tokenize
            tokens = self.tokenizer_.encode(
                combined_input,
                bos=True,
                eos=True,
                cutoff_len=self.config_.cutoff_len_
            )
            ret_tokens.append(tokens)

        end_idx = start_idx + len(ret_tokens)

        data_config = MLoRADataConfig(
            self.context_.name_,
            self.context_.type_,
            start_idx,
            end_idx,
            self._expand_batch_tokens,
            self._multitask_loss,
            self.task_name(),
        )

        return ret_tokens, [data_config] 