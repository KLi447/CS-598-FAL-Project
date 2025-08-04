import logging
from typing import Tuple

from transformers import AutoModel

from mlora.model.llm import LLMModel, QwenModel, LlamaModel, LlamaModel_TP
from mlora.model.tokenizer import Tokenizer

from huggingface_hub import login

import os

MODEL_TYPE_DICT = {
    "llama": LlamaModel,
    "qwen": QwenModel,
    "llama_tp": LlamaModel_TP
}

hf_token = os.getenv("HUGGING_FACE_TOKEN") 

logging.info(f"TOKEN: {hf_token}")

if hf_token:
    login(token=hf_token)


def load_partial_model(args) -> LLMModel:
    # load part of model to device
    assert args.rank != -1
    assert args.nodes >= args.rank

    logging.info(
        f"Pipeline parallelism, rank is {args.rank} and distributed over {args.nodes} nodes."
    )

    if "llama" in args.base_model.lower():
        model = LlamaModel.from_pretrained(
            path=args.base_model,
            device=args.device,
            precision=args.precision,
        )
    if "qwen" in args.base_model.lower():
        model = QwenModel.from_pretrained(
            path=args.base_model,
            device=args.device,
            precision=args.precision,
        )
    seq_model = model.sequential()
    num_layers = len(seq_model)

    balance = [num_layers // args.nodes] * args.nodes
    for i in range(num_layers % args.nodes):
        balance[i] += 1

    partial_model_to_device = [
        index + sum(balance[: args.rank])
        for index in range(0, balance[args.rank])
    ]

    del model

    return MODEL_TYPE_DICT[args.model_type].from_pretrained(
        path=args.base_model,
        device=args.device,
        precision=args.precision,
        partial_model_to_device=partial_model_to_device,
    )

def load_tp_model(args) -> LLMModel:
    assert args.rank != -1
    assert args.nodes > 0

    logging.info(
        f"Tensor parallelism, rank is {args.rank} and distributed over {args.nodes} nodes."
    )

    if args.model_type != "llama_tp":
        raise ValueError("Model type must be 'llama_tp' for tensor parallel loading.")

    return LlamaModel_TP.from_pretrained(
        path=args.base_model,
        device=args.device,
        precision=args.precision,
        rank=args.rank,
        world_size=args.nodes,
    )


def load_full_model(args) -> LLMModel:
    return MODEL_TYPE_DICT[args.model_type].from_pretrained(
        path=args.base_model,
        device=args.device,
        precision=args.precision,
        partial_model_to_device=None,
    )


def load_model(args) -> Tuple[Tokenizer, LLMModel]:
    assert args.precision in ["nf4", "fp4", "int8", "bf16", "fp16", "fp32"]
    assert args.model_type in MODEL_TYPE_DICT, f"unknown model type {args.model_type}"

    tokenizer = Tokenizer(args.base_model)

    if args.model_type == "qwen":  # FIXME
        tokenizer.bos_id_ = tokenizer.eos_id_

    if args.pipeline:
        model = load_partial_model(args)
    elif args.tensor_parallel:
        model = load_tp_model(args)
    else:
        model = load_full_model(args)

    if args.model_type == "qwen":  # FIXME
        model.pad_token_id_ = tokenizer.pad_id_

    return tokenizer, model
