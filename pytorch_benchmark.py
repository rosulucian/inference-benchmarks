import sys
import argparse
import pandas as pd

from utils import benchmark_pytorch

def parseargs(args):
    """
    Parse the arguments
    """
    parser = argparse.ArgumentParser(description='Simple script for benchmarking inference time on pytorch vision models')

    parser.add_argument('-f', '--framework', choices=['pytorch', 'onnx'], default='pytorch')
    parser.add_argument('-m', '--models', nargs='+', default=['alexnet'])
    parser.add_argument('-d', '--devices', nargs='+', default=['cpu'])
    parser.add_argument('-b', '--batch_size', nargs='+', default=[2**i for i in range(5)])
    parser.add_argument('-e', '--export', action='store_true', default=False)
    parser.add_argument('-v', '--verbose', action='store_true', default=False)

    return parser.parse_args(args)

def main(args=None):

    if args is None:
        args = sys.argv[1:]

    args = parseargs(args)
    args.size=224
        
    df = None

    if args.framework == 'pytorch':
        df = benchmark_pytorch(args)
    else:
        print(f'Framework {args.framework} not supported. Exiting..')

    if df is not None and args.export:
        df.to_csv('results.csv', index=None)

if __name__ == "__main__":

    main()
