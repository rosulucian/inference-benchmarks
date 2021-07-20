import torch
import time


def benchmark(model, devices, batch_size, verbose=False, size=224):
    inf_times = []

    if verbose:
        print(f'Running inference for size {size}x{size} for {model.__class__}')

    configs = [(device, bs) for device in devices for bs in batch_size]
    
    for device, bs in configs:
        model.eval()
        model.to(device)

        x = torch.rand(bs, 3, size, size).to(device)

        # add warmup -> move model/input to device
        model(x)

        start = time.time()
        with torch.no_grad():
            model(x)

        inf_time = (time.time() - start) * 1000 / bs

        if verbose:
            print(f'{device}-{bs}: {inf_time:.3f} ms')
        
        inf_times.append(inf_time)
        
    return inf_times