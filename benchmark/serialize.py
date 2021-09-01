import sys
import os
import argparse

from bench import benchmark
from utils import model_choices, cache_onnx_model, check_cache

def parseargs(args):
    """
    Parse the arguments
    """
    parser = argparse.ArgumentParser(description='Simple script for serializing or converting serialized models')

    parser.add_argument('-s', '--source', choices=['pytorch'], default='pytorch')
    parser.add_argument('-d', '--dir', action='store', type=str, default='models')
    parser.add_argument('-t', '--target', choices=['onnx', 'script'], default='onnx')
    parser.add_argument('-m', '--models', nargs='+', choices=model_choices+['all'], default=['alexnet'])

    parser.add_argument('-sz', '--size', action='store', type=int, default=224)

    return parser.parse_args(args)

def pytorch2onnx(models, cache_dir, input_size):
    cache_hit, not_cached = check_cache(models, cache_dir, 'onnx')

    if len(cache_hit) > 0:
        print(f'models: {cache_hit} already cached')

    for model in not_cached:
        cache_onnx_model(cache_dir, model, input_size)

def pytorch2script(models, cache_dir, input_size):
    pass

def main(args=None):

    if args is None:
        args = sys.argv[1:]

    args = parseargs(args)
    if 'all' in args.models:
        args.models = model_choices
        
    if args.source == 'pytorch':
        pytorch2onnx(args.models, args.dir, args.size)
    elif args.source == 'files':
        # TODO: onnx -> openvino
        pass
    else:
        print(f'Cant convert from {args.source}')
        pass


if __name__ == "__main__":

    main()
