import os
import torch
import torchvision

def check_cache(models, cache_dir, ext):
    cached = [f.split('.')[0] for f in os.listdir(cache_dir) if ext in f]

    cache_hit = [m for m in models if m in cached]
    not_cached = [m for m in models if m not in cached]

    return cache_hit, not_cached

def is_cached(model_name, cache_dir, ext):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    onnx_file = f'{cache_dir}/{model_name}.{ext}'

    if os.path.isfile(onnx_file):
        return onnx_file
    else:
        return None

def cache_onnx_model(cache_dir, model_name, size):
    dummy_input = torch.rand(1, 3, size, size)

    model = getattr(torchvision.models, model_name)
    model = model(pretrained=False).eval()
    
    model_file = f'{cache_dir}/{model_name}.onnx'

    torch.onnx.export(
        model, 
        dummy_input, 
        model_file,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input' : {0 : 'batch_size'}, 
                    'output' : {0 : 'batch_size'},}
        )

    return model_file

def cache_script_model(cache_dir, model_name, size):
    dummy_input = torch.rand(1, 3, size, size)

    model = getattr(torchvision.models, model_name)
    model = model(pretrained=False).eval()
    
    model_file = f'{cache_dir}/{model_name}.script'

    traced_script_module = torch.jit.trace(model, dummy_input)
    model_file = traced_script_module.save(mode_file)

    return model_file 

def onnx_model(model_name, cache_dir='models', size=224):

    onnx_file = is_cached(model_name, cache_dir, 'onnx')
    if onnx_file is not None:
        return onnx_file

    onnx_file = cache_onnx_model(cache_dir, model_name, size)
    
    return onnx_file

def export_results(df, args, base_dir='results'):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    file_name = args.name if args.name is not None else args.framework
    
    df.to_csv(f'results/{args.file_name}.csv', index=None)


model_choices = [
    'alexnet',
    'resnet18',
    'resnet50',
    'vgg16',
    'vgg19',
    'squeezenet1_1',
    'densenet121',
    'inception_v3',
    'googlenet',
    # shufflenet not convertible to onnx; 
    # 'shufflenet_v2_x1_0',
    'mobilenet_v2',
    'mobilenet_v3_large',
    'mobilenet_v3_small',
    'resnext50_32x4d',
    'wide_resnet50_2',
    'mnasnet0_5',
    ]

framework_choices = [
    'pytorch', 
    'script', 
    'onnx',
    # 'openvino',
    ]
