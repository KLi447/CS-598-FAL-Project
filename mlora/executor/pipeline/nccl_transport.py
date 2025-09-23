import os
import torch
import torch.distributed as dist
import pickle
from typing import Optional, Dict, Tuple, List
import queue
from threading import Thread
import time
import logging

class PipeMessageType:
    TENSOR = "TENSOR"
    COMM = "COMM"

class PipeMessage:
    def __init__(self, msg_type: str, tensor: Optional[torch.Tensor] = None,
                 comm_data=None, meta_tensor: Optional[torch.Tensor] = None):
        self.msg_type_ = msg_type
        self.tensor_data_ = tensor
        self.comm_data_ = comm_data
        self.meta_tensor_ = meta_tensor

class NcclTransport:
    def __init__(self, rank: int, world_size: int, device: torch.device, header_queue = None):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self._shutdown = False
        self.header_queue = header_queue or queue.Queue()

        # Set default env if not provided
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "12355"

        if not dist.is_initialized():
            dist.init_process_group(
                backend="gloo",
                init_method="env://",
                rank=rank,
                world_size=world_size,
            )

        self.prev_rank: Optional[int] = rank - 1 if rank > 0 else None
        self.next_rank: Optional[int] = rank + 1 if rank < world_size - 1 else None

        self.nccl_groups: Dict[Tuple[int, int], dist.ProcessGroup] = {}
        if self.prev_rank is not None:
            pair = (self.prev_rank, self.rank)
            self.nccl_groups[pair] = dist.new_group(ranks=list(pair), backend="nccl")
        if self.next_rank is not None:
            pair = (self.rank, self.next_rank)
            self.nccl_groups[pair] = dist.new_group(ranks=list(pair), backend="nccl")

        dist.barrier()
        print(f"[Rank {self.rank}] NcclTransport ready. Neighbors: prev={self.prev_rank}, next={self.next_rank}")

        self._listener_threads = []
        if self.prev_rank is not None:
            t = Thread(target=self._header_listener_for_src, args=(self.prev_rank,), daemon=True)
            t.start()
            self._listener_threads.append(t)
        if self.next_rank is not None:
            t = Thread(target=self._header_listener_for_src, args=(self.next_rank,), daemon=True)
            t.start()
            self._listener_threads.append(t)

    def _header_listener_for_src(self, src: int):
        while not self._shutdown:
            try:
                hdr = self._recv_header(src)   # will block
                self.header_queue.put((src, hdr))
                print(f"[Rank {self.rank}] header_listener enqueued from {src}: {hdr}", flush=True)
                print(f"Queue size: {self.header_queue.qsize()}")
            except Exception as e:
                print(f"[Rank {self.rank}] header_listener error for src {src}: {e}", flush=True)
                time.sleep(0.5)

    def _send_header(self, header: dict, dst: int):
        b = pickle.dumps(header)
        n = torch.tensor([len(b)], dtype=torch.int64, device="cpu")
        dist.send(n, dst=dst)
        buf = torch.frombuffer(b, dtype=torch.uint8)
        dist.send(buf, dst=dst)

    def _recv_header(self, src: int) -> dict:
        n = torch.empty(1, dtype=torch.int64, device="cpu")
        dist.recv(n, src=src)
        length = int(n.item())
        buf = torch.empty(length, dtype=torch.uint8, device="cpu")
        dist.recv(buf, src=src)
        return pickle.loads(buf.numpy().tobytes())

    def send_message(self, msg: PipeMessage, neighbor: str):
        if neighbor == "next":
            dst = self.next_rank
        elif neighbor == "prev":
            dst = self.prev_rank
        else:
            raise ValueError(f"Bad neighbor: {neighbor}")
        if dst is None:
            raise RuntimeError(f"[Rank {self.rank}] No {neighbor} neighbor to send to.")

        if msg.tensor_data_ is not None:
            header = {
                "msg_type": PipeMessageType.TENSOR,
                "shape": list(msg.tensor_data_.shape),
                "dtype": str(msg.tensor_data_.dtype),
            }
            if getattr(msg, "meta_tensor_", None) is not None:
                header["meta_shape"] = list(msg.meta_tensor_.shape)
                header["meta_dtype"] = str(msg.meta_tensor_.dtype)
            if getattr(msg, "comm_data_", None) is not None:
                header["comm_data"] = msg.comm_data_
            self._send_header(header, dst)

            payload = msg.tensor_data_.contiguous().to(self.device, non_blocking=True)
            group = self._pair_group(self.rank, dst)
            dist.broadcast(payload, src=self.rank, group=group)

            if getattr(msg, "meta_tensor_", None) is not None:
                meta_cuda = msg.meta_tensor_.to(self.device, non_blocking=True).contiguous()
                dist.broadcast(meta_cuda, src=self.rank, group=group)

            print(f"[Rank {self.rank}] Sent tensor to {neighbor}")
        else:
            header = {
                "msg_type": PipeMessageType.COMM,
                "comm_data": msg.comm_data_,
            }
            self._send_header(header, dst)
            print(f"[Rank {self.rank}] Sent comm message to {neighbor}")

    def recv_message(self, neighbor: str, block: bool = False, only_type: Optional[str] = None) -> Optional[PipeMessage]:
        if neighbor == "next":
            expected_src = self.next_rank
        elif neighbor == "prev":
            expected_src = self.prev_rank
        else:
            raise ValueError(f"Bad neighbor: {neighbor}")

        if expected_src is None:
            return None

        temp_store = []
        found_item = None

        while True:
            try:
                src_rank, header = self.header_queue.get(
                    block=block, timeout=0.1 if block else 0
                )
            except queue.Empty:
                for item in temp_store:
                    self.header_queue.put(item)
                return None

            if src_rank != expected_src:
                temp_store.append((src_rank, header))
                if not block:
                    break
                continue

            if only_type and header.get("msg_type") != only_type:
                temp_store.append((src_rank, header))
                if not block:
                    break
                continue

            found_item = (src_rank, header)
            break

        for item in temp_store:
            self.header_queue.put(item)

        if found_item is None:
            return None

        src_rank, header = found_item
        msg_type = header.get("msg_type")

        if msg_type == PipeMessageType.TENSOR:
            shape = tuple(header["shape"])
            dtype = getattr(torch, header["dtype"].split(".")[-1])
            payload = torch.empty(shape, dtype=dtype, device=self.device)

            group = self._pair_group(self.rank, src_rank)
            dist.broadcast(payload, src=src_rank, group=group)

            meta_payload = None
            if "meta_shape" in header:
                meta_shape = tuple(header["meta_shape"])
                meta_dtype = getattr(torch, header["meta_dtype"].split(".")[-1])
                meta_payload = torch.empty(meta_shape, dtype=meta_dtype, device=self.device)
                dist.broadcast(meta_payload, src=src_rank, group=group)

            return PipeMessage(
                msg_type=msg_type,
                tensor=payload,
                comm_data=header.get("comm_data"),
                meta_tensor=meta_payload,
            )

        elif msg_type == PipeMessageType.COMM:
            return PipeMessage(
                msg_type=msg_type,
                tensor=None,
                comm_data=header.get("comm_data"),
                meta_tensor=None,
            )

        else:
            return None

    def _pair_group(self, a: int, b: int):
        key = (a, b) if (a, b) in self.nccl_groups else (b, a)
        return self.nccl_groups[key]

    def stop(self):
        self._shutdown = True
        dist.barrier()
        dist.destroy_process_group()
        print(f"[Rank {self.rank}] Transport stopped.")
