import torch
import gc

# Memory Report
## Analyzes model memory usage
def analyze_model_memory(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total_params - trainable
    param_mem_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    buffer_mem_mb = sum(b.numel() * b.element_size() for b in model.buffers()) / 1024**2
    print()
    print("MODEL MEMORY ANALYSIS")
    print()
    print(f"Total parameters:           {total_params:,}")
    print(f"  Trainable params:         {trainable:,}")
    print(f"  Non-trainable params:     {non_trainable:,}")
    print(f"Parameter memory footprint: {param_mem_mb:.2f} MB")
    print(f"Buffer memory footprint:    {buffer_mem_mb:.2f} MB")
    print(f"Total model memory:         {param_mem_mb + buffer_mem_mb:.2f} MB")

    print("\nLayer-wise parameter counts:")
    for name, module in model.named_children():
        layer_params = sum(p.numel() for p in module.parameters())
        layer_mem   = sum(p.numel() * p.element_size() for p in module.parameters()) / 1024**2
        print(f"  {name:<30} {layer_params:>12,} params ({layer_mem:.2f} MB)")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("\nGPU Memory Status:")
        for i in range(torch.cuda.device_count()):
            props     = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**2
            reserved  = torch.cuda.memory_reserved(i)  / 1024**2
            print(f"  GPU {i}: {props.name}")
            print(f"    Allocated: {allocated:.1f} MB")
            print(f"    Reserved : {reserved:.1f} MB")
            print(f"    Total    : {props.total_memory/1024**2:.1f} MB")
            print(f"    Free     : {(props.total_memory - reserved)/1024**2:.1f} MB")

## Displays proceeding data structure by showing information about tensor shapes, dict contents, and data types
def debug_oom_sample(sample_proc):
    print(">>> Proceeding dict structure\n")
    for k, v in sample_proc.items():
        if isinstance(v, torch.Tensor):
            print(f"{k:>25}: Tensor shape={tuple(v.shape)}, dtype={v.dtype}")
        elif isinstance(v, dict):
            keys = list(v.keys())[:5]
            more = "…" if len(v)>5 else ""
            print(f"{k:>25}: dict with keys {keys}{more}")
        elif isinstance(v, list):
            print(f"{k:>25}: list (len={len(v)})")
        else:
            s = str(v)
            snippet = (s[:50] + "…") if len(s)>50 else s
            print(f"{k:>25}: {type(v).__name__} = {snippet}")
    print("\n" + "-"*60 + "\n")