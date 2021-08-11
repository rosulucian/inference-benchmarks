import os
import torch
import torchvision

def check_onnx_file(base_dir, model_name):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    onnx_file = f'{base_dir}/{model_name}.onnx'

    if os.path.isfile(onnx_file):
        return onnx_file
    else:
        return None

def onnx_model(model_name, base_dir='models', size=224):

    onnx_file = check_onnx_file(base_dir, model_name)
    if onnx_file is not None:
        return onnx_file

    dummy_input = torch.rand(1, 3, size, size)

    model = getattr(torchvision.models, model_name)
    model = model(pretrained=False).eval()
    
    onnx_file = f'{base_dir}/{model_name}.onnx'

    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_file,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input' : {0 : 'batch_size'}, 
                    'output' : {0 : 'batch_size'},}
        )
    
    return onnx_file

def export_results(df, args, base_dir='results'):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    df.to_csv(f'results/{args.framework}.csv', index=None)


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
    'jit', 
    'onnx',
    'openvino',
    ]
