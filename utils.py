import torch
import time


def benchmark(model, device, batch_sizes, verbose=False, size=224):
    inf_times, batch_size  = [], []
    
    for bs in batch_sizes:
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
        batch_size.append(bs)
        
    return inf_times, batch_size