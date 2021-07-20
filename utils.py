import pandas as pd
import time
import torch
import torchvision

def benchmark_pytorch(args):
    inf_times, batch_sizes, models, devices = [], [], [], []

    print(f'torch: {torch.__version__}')
    print(f'torchvision: {torchvision.__version__}')

    for model_name in args.models:
        if args.verbose:
            print(f'Running inference for size {args.size}x{args.size} for {model_name}')
            
        try:
            model = getattr(torchvision.models, model_name)
            model = model(pretrained=False)

            for device in args.devices:
                t = benchmark(model, device, args.batch_size, size=args.size, verbose=args.verbose)
                devices += [device] * len(args.batch_size)

                inf_times += t
                batch_sizes += args.batch_size
                models += [model_name] * len(args.batch_size)

        except AssertionError as error:
            print(error)

    df = pd.DataFrame(list(zip(models, devices, batch_sizes, inf_times)), columns=['model', 'device', 'bs', 'time'])
    df['engine'] = 'pytorch'

    return df

def benchmark(model, device, batch_sizes, verbose=False, size=224):
    inf_times  = []
    
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
        
    return inf_times
    