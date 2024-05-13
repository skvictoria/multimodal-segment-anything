import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from segment_anything import SamPredictor, sam_model_registry

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"Rank {rank} is set to use GPU {torch.cuda.current_device()} with {torch.cuda.get_device_properties(rank).total_memory} bytes of memory")

if __name__ == "__main__":
    rank = int(os.getenv('LOCAL_RANK', 0))
    world_size = torch.cuda.device_count()
    setup(rank, world_size)

    try:
        model = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth").to(rank)
        model = DDP(model, device_ids=[rank])
        print(f"Model successfully loaded on GPU {rank}")
    except Exception as e:
        print(f"Failed to load model on GPU {rank}: {str(e)}")