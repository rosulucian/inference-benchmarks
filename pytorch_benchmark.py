import torch
import sys
import argparse
from torch._C import device
import torchvision
from utils import benchmark

def parseargs(args):
    """
    Parse the arguments
    """
    parser = argparse.ArgumentParser(description='Simple script for benchmarking inference time on pytorch vision models')

    parser.add_argument('-m', '--models', nargs='+', default=['alexnet'])
    parser.add_argument('-d', '--devices', nargs='+', default=['cpu'])
    parser.add_argument('-b', '--batch_size', nargs='+', default=[2**i for i in range(5)])
    parser.add_argument('-v', '--verbose', action='store_true', default=False)

    return parser.parse_args(args)

def benchmark_model(model_name, devices, batch_size, verbose):
    try:
        model = getattr(torchvision.models, model_name)
        model = model(pretrained=False)

        benchmark(model, devices, batch_size, verbose=verbose)

    except AssertionError as error:
        print(error)

def main(args=None):

    if args is None:
        args = sys.argv[1:]
    args = parseargs(args)

    print(f'torch: {torch.__version__}')
    print(f'torchvision: {torchvision.__version__}')

    for model in args.models:
        benchmark_model(model, args.devices, args.batch_size, verbose=args.verbose)

if __name__ == "__main__":

    main()
