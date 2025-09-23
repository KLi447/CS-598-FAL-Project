import logging
import torch
from mlora.model.args import ModelData
from typing import Any
from .nccl_transport import NcclTransport, PipeMessage, PipeMessageType


class RecvOperator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dummy: torch.Tensor, transport, message: PipeMessage, role: str):
        if message is None:
            raise RuntimeError("RecvOperator.forward called with message=None")
        assert message.msg_type_ == PipeMessageType.TENSOR, \
            "RecvOperator.forward expected a TENSOR message"

        ctx.transport = transport
        ctx.role = role
        ctx.meta_tensor = getattr(message, "meta_tensor_", None)
        ctx.comm_data = getattr(message, "comm_data_", None)

        out = message.tensor_data_
        if not out.requires_grad:
            out = out.requires_grad_(True)

        if role == "tail":
            out.retain_grad()

            def _grad_hook(grad: torch.Tensor):
                grad_msg = PipeMessage(
                    PipeMessageType.TENSOR,
                    tensor=grad.detach(),
                    meta_tensor=ctx.meta_tensor,
                    comm_data=ctx.comm_data,
                )
                ctx.transport.send_message(grad_msg, "prev")
                logging.info("[Tail] Sent initial gradient upstream.")

            out.register_hook(_grad_hook)
            return out
    
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.role == "tail":
            return None, None, None, None

        grad_payload = grad_output.detach()
        grad_msg = PipeMessage(
            PipeMessageType.TENSOR,
            tensor=grad_payload,
            meta_tensor=getattr(ctx, "meta_tensor", None),
            comm_data=getattr(ctx, "comm_data", None),
        )
        ctx.transport.send_message(grad_msg, "prev")
        logging.info(f"[{ctx.role.capitalize()}] Sent gradient upstream.")
        return None, None, None, None


class SendOperator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dummy: torch.Tensor, transport, message: PipeMessage, role: str):
        if message is None:
            raise RuntimeError("SendOperator.forward called with message=None")

        ctx.transport = transport
        ctx.role = role
        ctx.meta_tensor = getattr(message, "meta_tensor_", None)
        ctx.comm_data = getattr(message, "comm_data_", None)

        if role == "tail":
            return message.tensor_data_

        transport.send_message(message, "next")

        device = getattr(transport, "device", None) or torch.device("cpu")
        phony = torch.tensor(1.0, requires_grad=True, device=device)
        return phony

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return None, None, None, None
