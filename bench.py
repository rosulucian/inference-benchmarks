import abc
import time
import torch
import torchvision

import pandas as pd
import numpy as np
import onnxruntime as ort

from utils import onnx_model

class BenchModel(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def forward(self, x, device, warmup=False):
        pass

    @abc.abstractmethod
    def prepare(self, device, args):
        pass

    @abc.abstractmethod
    def get_dummy_input(self):
        pass

    def benchmark(self, device, args, warmup=False):
        self.prepare(device, args)
        
        inf_times  = []
    
        for bs in args.batch_size:
            x = self.get_dummy_input(bs, device)

            inf_time = self.forward(x, device, warmup) / bs

            inf_times.append(inf_time)

            if args.verbose:
                print(f'{device}-{bs}: {inf_time:.3f} ms')
            
        return inf_times

class Pytorch_bench(BenchModel):
    def __init__(self, model_name, args):
        model = getattr(torchvision.models, model_name)
        model = model(pretrained=False)
        model.eval()

        self.model = model
        self.size = args.size

    def prepare(self, device, args):
        pass

    def forward(self, x, device, warmup=False):
        self.model.to(device)

        if warmup is True:
            self.model(x)

        start = time.time()
        with torch.no_grad():
            self.model(x)

        inf_time = (time.time() - start) * 1000

        return inf_time

    def get_dummy_input(self, bs, device):
        dummy = torch.rand(bs, 3, self.size, self.size).to(device)

        return dummy

class Onnx_bench(BenchModel):
    def __init__(self, model_name, args):
        self.onnx_file = onnx_model(model_name)
        self.size = args.size

    def prepare(self, device, args):
        provider = 'CPUExecutionProvider' if device == 'cpu' else 'CUDAExecutionProvider'
        self.session = ort.InferenceSession(self.onnx_file, providers=[provider])

    def forward(self, x, device, warmup=False):
        
        if warmup is True:
            self.session.run(None, {'input': x})

        start = time.time()
        with torch.no_grad():
            self.session.run(None, {'input': x})

        inf_time = (time.time() - start) * 1000

        return inf_time

    def get_dummy_input(self, bs, device):
        dummy = np.random.rand(bs, 3, 224, 224).astype(np.float32)

        return dummy

def get_model(model_name, args):
    model = None

    if args.framework == 'pytorch':
        model = Pytorch_bench(model_name, args)
    elif args.framework == 'jit':
        pass
    elif args.framework == 'onnx':
        model = Onnx_bench(model_name, args)

    return model

def benchmark(args):
    inf_times, batch_sizes, models, devices = [], [], [], []

    print(f'torch: {torch.__version__}')
    print(f'torchvision: {torchvision.__version__}')
    if args.framework == 'onnx':
        print(f'onnx rutime: {ort.__version__}')

    for model_name in args.models:
        if args.verbose:
            print(f'Running inference on {model_name}, size: {args.size}x{args.size}')
            
        try:
            model = get_model(model_name, args)

            for device in args.devices:
                t = model.benchmark(device, args, warmup=True)

                inf_times += t
                batch_sizes += args.batch_size
                devices += [device] * len(args.batch_size)
                models += [model_name] * len(args.batch_size)

            del model

        except AssertionError as error:
            print(error)

    df = pd.DataFrame(list(zip(models, devices, batch_sizes, inf_times)), columns=['model', 'device', 'bs', 'time'])
    df['engine'] = args.framework

    return df
