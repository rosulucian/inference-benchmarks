import torch
import sys
import argparse
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

def main(args=None):

    if args is None:
        args = sys.argv[1:]
    args = parseargs(args)

    size=224

    print(f'torch: {torch.__version__}')
    print(f'torchvision: {torchvision.__version__}')

    for model_name in args.models:
        if args.verbose:
            print(f'Running inference for size {size}x{size} for {model_name}')
            
        try:
            model = getattr(torchvision.models, model_name)
            model = model(pretrained=False)

            for device in args.devices:
                inf_times, batch_size = benchmark(model, device, args.batch_size, size=size, verbose=args.verbose)

        except AssertionError as error:
            print(error)


if __name__ == "__main__":

    main()
