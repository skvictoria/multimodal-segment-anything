import torch
import torch.distributed as dist
import os

def check_device():
    rank = int(os.environ['LOCAL_RANK'])
    print(f"Rank: {rank}, Current GPU: {torch.cuda.current_device()}, Available GPUs: {torch.cuda.device_count()}")
    torch.cuda.set_device(rank)
    print(f"Using GPU: {torch.cuda.current_device()} for rank {rank}")
    # Add a simple computation to test GPU
    a = torch.rand(1000, 1000, device="cuda")
    b = torch.rand(1000, 1000, device="cuda")
    c = a @ b

def main():
    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    tensor = torch.rand(10).cuda()
    dist.all_reduce(tensor)
    print(f"Rank {rank}: {tensor}")

if __name__ == "__main__":
    #main()
    check_device()

