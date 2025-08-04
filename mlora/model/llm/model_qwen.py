import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, override

import torch
from torch.nn.modules import Sequential
from transformers import AutoConfig, AutoModelForCausalLM

from mlora.model.args import LinearInfo, LLMModelArgs, Masks, ModelData
from mlora.model.checkpoint import CheckpointRecomputeFunction
from mlora.model.modules import AdapterModel, Decoder, Embedding, OutputLayer, RMSNorm
from mlora.profiler import nvtx_wrapper, set_backward_tracepoint
from mlora.utils import is_package_available

if is_package_available("bitsandbytes"):
    from transformers import BitsAndBytesConfig
else:
    from mlora.utils import BitsAndBytesConfig

from .model_llm import LLMModel

from accelerate import init_empty_weights, infer_auto_device_map
from accelerate.utils import get_balanced_memory


# input_tokens shape is: batch_size * seq_len
#   default: upper triangular matrix like below, i.e. diagonal = 1
#            0 -inf -inf
#            0    0 -inf
#            0    0    0
# additional_mask: batch_size * seq_len
#   default: is None the matrix like default, if set true, the mask metric will be -inf
#   example: [[True, False, False]]
#           -inf -inf -inf
#           -inf    0 -inf
#           -inf    0    0
def precompute_mask(
    input_tokens: torch.Tensor,
    n_heads: int,
    device: str,
    additional_mask: List[Masks] | None = None,
    diagonal: int = 1,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if input_tokens.dim() == 2:
        batch_size, seq_len = input_tokens.shape
    elif input_tokens.dim() == 3:
        batch_size, seq_len, _ = input_tokens.shape
    else:
        raise Exception(f"input dim is not correct {input_tokens.dim()}")

    TORCH_MIN_VALUE = torch.finfo(dtype).min
    mask = torch.full(
        (batch_size, n_heads, seq_len, seq_len),
        TORCH_MIN_VALUE,
        device=device,
        dtype=dtype,
    )
    mask = torch.triu(mask, diagonal=diagonal)

    if additional_mask is not None:
        masks_metric = torch.tensor(additional_mask, dtype=torch.bool, device=device)
        masks_metric = masks_metric.view(batch_size, 1, 1, seq_len)
        masks_metric = masks_metric.expand(-1, n_heads, seq_len, -1)
        mask = torch.masked_fill(mask, masks_metric, TORCH_MIN_VALUE)

    mask.requires_grad_(False)

    return mask.to(device=device, dtype=dtype)


QwenSequentialModuleIO = Tuple[
    torch.Tensor,
    torch.Tensor,
    ModelData,
    bool,
]
LEN_QWEN_SEQUENTIAL_MODULE_IO = 4

QwenCompatibleModelTypes = ["qwen2", "qwen3"]


class QwenSequentialWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.wrapper_module_ = module

    def name(self) -> str:
        return type(self.wrapper_module_).__name__

    def forward(self, input: QwenSequentialModuleIO) -> QwenSequentialModuleIO:
        assert len(input) == LEN_QWEN_SEQUENTIAL_MODULE_IO
        assert isinstance(input[0], torch.Tensor)
        assert isinstance(input[1], torch.Tensor)
        assert isinstance(input[2], ModelData)
        assert isinstance(input[3], bool)

        # auto catch the input argument
        @nvtx_wrapper("f_embedding")
        def embedding_forward():
            output = self.wrapper_module_.forward(input[0])
            if input[-1]:
                output = output.requires_grad_(True)
            return (output,) + input[1:]

        def decoder_forward():
            if input[-1]:
                output = CheckpointRecomputeFunction(
                    self.wrapper_module_.forward, *input[:-1]
                )
                set_backward_tracepoint(output.grad_fn, "b_checkpoint")
            else:
                output = self.wrapper_module_.forward(*input[:-1])
            return (output,) + input[1:]

        @nvtx_wrapper("f_rmsnorm")
        def rmsnorm_forward():
            output = self.wrapper_module_.forward(input[0])
            set_backward_tracepoint(output.grad_fn, "b_rmsnorm")
            return (output,) + input[1:]

        @nvtx_wrapper("f_output")
        def output_layer_forward():
            output = self.wrapper_module_.forward(input[0])
            set_backward_tracepoint(output.grad_fn, "b_output")
            return (output,) + input[1:]

        forward_func_dict = {
            "Embedding": embedding_forward,
            "Decoder": decoder_forward,
            "RMSNorm": rmsnorm_forward,
            "OutputLayer": output_layer_forward,
        }

        module_name = self.name()
        assert (
            module_name in forward_func_dict
        ), f"error module name {module_name}"

        return forward_func_dict[module_name]()


class QwenModel(LLMModel):
    seq_module_: torch.nn.Sequential

    def __init__(self, args: LLMModelArgs):
        super().__init__()
        self.name_or_path_: str = args.name_or_path_
        self.norm_eps_ = args.norm_eps_
        self.device_ = args.device_
        self.n_heads_ = args.n_heads_
        self.dim_ = args.dim_
        self.vocab_size_ = args.vocab_size_
        self.pad_token_id_ = args.pad_token_id_
        self.bos_token_id_ = args.bos_token_id_
        self.eos_token_id_ = args.eos_token_id_

    @override
    def forward(self, input: ModelData) -> torch.Tensor:
        tokens = torch.tensor(
            input.batch_tokens_, dtype=torch.int64, device=self.device_
        )
        mask = precompute_mask(tokens, self.n_heads_, self.device_, input.batch_mask_)

        data = (tokens, mask, input, bool(input.enable_checkpoint_))

        for seq_layer in self.seq_module_:
            data = seq_layer.forward(data)

        return data[0]

    @override
    @staticmethod
    def from_pretrained(
        path: str,
        device: str,
        precision: str,
        partial_model_to_device: Optional[List[int]] = None,
    ) -> LLMModel:
        def create_device_map() -> str | Dict[str, str]:
            device_map: str | Dict[str, str]
            if partial_model_to_device is None:
                device_map = device
            else:
                config = AutoConfig.from_pretrained(path)
                # Be careful, this is hard coded.
                weight_map = [
                    "model.embed_tokens",
                    "model.rotary_emb",
                    *[
                        f"model.layers.{layer_id}"
                        for layer_id in range(0, config.num_hidden_layers)
                    ],
                    "model.norm",
                    "lm_head",
                ]
                device_map = {map_item: "disk" for map_item in weight_map}
                for partial_weight in partial_model_to_device:
                    device_map[weight_map[partial_weight]] = device
            return device_map

        load_type_dict = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }

        additional_load_args = {
            "device_map": create_device_map(),
            "torch_dtype": torch.float32,
        }

        logging.info(f"Loading model with precision - {precision}")

        if precision in load_type_dict:
            additional_load_args["torch_dtype"] = load_type_dict[precision]
        else:
            load_4bit = precision in ["nf4", "fp4"]
            load_8bit = precision == "int8"
            
            additional_load_args["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=load_4bit,
                load_in_8bit=load_8bit,
                llm_int8_enable_fp32_cpu_offload=True,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type=precision,
            )

        qwen_model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, **additional_load_args)

        if qwen_model.config.model_type not in QwenCompatibleModelTypes:
            raise AssertionError(f"Unsupported model type {qwen_model.config.model_type}. Expected one of {QwenCompatibleModelTypes}.")

        logging.info(f"Loading Qwen compatible model - {qwen_model.config.model_type}")

        qwen_args = LLMModelArgs(qwen_model.config)
        if qwen_args.pad_token_id_ is None:
            qwen_args.pad_token_id_ = -1
        qwen_args.device_ = device
        qwen_args.dtype_ = qwen_model.dtype

        model = QwenModel.convert_model_from_huggingface(qwen_model, qwen_args)

        return model

    @staticmethod
    def convert_model_from_huggingface(
        qwen_model: AutoModelForCausalLM, qwen_args: LLMModelArgs
    ):
        qwen_model.requires_grad_(False)
        seq_model: OrderedDict[str, torch.nn.Module] = OrderedDict()

        seq_model["embedding"] = QwenSequentialWrapper(
            Embedding(qwen_model.model.embed_tokens.weight, qwen_args.pad_token_id_)
        )

        for idx, target_layer in enumerate(qwen_model.model.layers):
            decoder = Decoder(idx, qwen_args)
            decoder.from_pretrained(target_layer, qwen_args.norm_eps_)
            seq_model[f"layer{idx}"] = QwenSequentialWrapper(decoder)

        seq_model["norm"] = QwenSequentialWrapper(
            RMSNorm(qwen_model.model.norm.weight, qwen_args.norm_eps_)
        )

        seq_model["output"] = QwenSequentialWrapper(
            OutputLayer(qwen_model.lm_head.weight, qwen_args)
        )

        model = QwenModel(qwen_args)
        model.seq_module_ = torch.nn.Sequential(seq_model)
        return model

    @override
    def load_adapter(self, adapter_model: AdapterModel):
        for module in self.seq_module_:
            if isinstance(module, QwenSequentialWrapper) and module.name() == "Decoder":
                module.wrapper_module_.load_adapter(adapter_model)

    @override
    def offload_adapter(self, adapter_name: str):
        for module in self.seq_module_:
            if isinstance(module, QwenSequentialWrapper) and module.name() == "Decoder":
                module.wrapper_module_.offload_adapter(adapter_name)

    @override
    def linears_info(self) -> OrderedDict[str, LinearInfo]:
        ret_val = OrderedDict()
        for module in self.seq_module_:
            if isinstance(module, QwenSequentialWrapper) and module.name() == "Decoder":
                ret_val.update(module.wrapper_module_.linears_info())
        return ret_val

    @override
    def sequential(self) -> Sequential:
        return self.seq_module_