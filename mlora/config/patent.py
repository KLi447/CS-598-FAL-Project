from dataclasses import dataclass
from typing import Optional

from .adapter import AdapterConfig
from .dataset import DatasetConfig
from .task import TaskConfig


@dataclass
class PatentTaskConfig(TaskConfig):
    type_: str = "patent"
    name_: str = "patent_classification_task"
    adapter_: AdapterConfig = None
    dataset_: Optional[DatasetConfig] = None 