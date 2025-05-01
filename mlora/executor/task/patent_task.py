import logging
from collections import OrderedDict
from typing import Dict, List, override

from datasets import load_dataset
from tqdm import tqdm

from mlora.config import TaskConfig
from mlora.model.args import LinearInfo, Tokens
from mlora.model.tokenizer import Tokenizer
from mlora.prompter import Prompter, PrompterFactory
from mlora.executor.task.task import Task

class PatentTask(Task):
    def _pre_dataset(self):
        preprocess_func: Dict[str, Callable] = {
            "default": lambda data: data,
            "shuffle": lambda data: self._shuffle_data(data),
            "sort": lambda data: data.sort(),
        }

        if self.config_.dataset_ is None:
            logging.info(
                "Task dataset is empty, maybe in pipeline we do not load dataset."
            )
            return

        self.prompter_ = PrompterFactory.create(self.config_.dataset_)

        logging.info(f"Task load data from {self.config_.dataset_.data_path_}")
        
        # Load dataset from Hugging Face
        data = load_dataset(
            self.config_.dataset_.data_path_,
            self.config_.dataset_.subset,
            split="train"
        )

        preprocess_type = self.config_.dataset_.preprocess_
        if preprocess_type not in preprocess_func:
            raise NotImplementedError

        # Process data according to the data preprocess_type
        data = preprocess_func[preprocess_type](data)
        logging.info(
            f"Adapter {self.config_.adapter_.name_} "
            f"data size: {len(data)}"
        )

        # Convert to the format expected by mLoRA
        for data_point in tqdm(data):
            formatted_data = {
                "text": data_point[self.config_.dataset_.text_column],
                "label": data_point[self.config_.dataset_.label_column]
            }
            self.data_.append(formatted_data)

    @override
    def prepare(self, linears_info: OrderedDict[str, LinearInfo], tokenizer: Tokenizer):
        self.tokenizer_ = tokenizer
        self._pre_dataset()
        self._pre_context(linears_info) 