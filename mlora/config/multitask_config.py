from typing import Dict, Mapping
from mlora.config import TrainTaskConfig, AdapterConfig, DatasetConfig

class MultitaskConfig(TrainTaskConfig):
    """Configuration for multitask learning."""
    
    lm_weight_: float
    cls_weight_: float
    contrastive_weight_: float
    temperature_: float

    __params_map: Dict[str, str] = {
        "lm_weight": "lm_weight_",
        "cls_weight": "cls_weight_",
        "contrastive_weight": "contrastive_weight_",
        "temperature": "temperature_",
    }

    def __init__(
        self,
        config: Dict[str, str],
        adapters: Mapping[str, AdapterConfig],
        datasets: Mapping[str, DatasetConfig],
    ):
        super().__init__(config, adapters, datasets)
        self.init(self.__params_map, config)
        
        # Convert string values to float
        self.lm_weight_ = float(self.lm_weight_)
        self.cls_weight_ = float(self.cls_weight_)
        self.contrastive_weight_ = float(self.contrastive_weight_)
        self.temperature_ = float(self.temperature_) 