import sys
import argparse

from bench import benchmark
from utils import export_results, model_choices

def parseargs(args):
    """
    Parse the arguments
    """
    parser = argparse.ArgumentParser(description='Simple script for benchmarking inference time on pytorch vision models')

    parser.add_argument('-f', '--framework', choices=['pytorch', 'jit', 'onnx'], default='pytorch')
    parser.add_argument('-m', '--models', nargs='+', choices=model_choices, default=['alexnet'])
    parser.add_argument('-d', '--devices', nargs='+', choices=['cuda', 'cpu'], default=['cpu'])
    parser.add_argument('-b', '--batch_size', nargs='+', default=[2**i for i in range(7)])
    parser.add_argument('-s', '--size', action='store', default=224)
    parser.add_argument('-e', '--export', action='store_true', default=False)
    parser.add_argument('-v', '--verbose', action='store_true', default=False)

    return parser.parse_args(args)

def main(args=None):

    if args is None:
        args = sys.argv[1:]

    args = parseargs(args)
        
    df = benchmark(args)

    if df is not None and args.export:
        export_results(df, args)

if __name__ == "__main__":

    main()
