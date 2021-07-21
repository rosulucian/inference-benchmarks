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

def onnx_model(model_name, base_dir='onnx', size=224):

    onnx_file = check_onnx_file(base_dir, model_name)
    if onnx_file is not None:
        return onnx_file

    dummy_input = torch.rand(1, 3, size, size)

    model = getattr(torchvision.models, model_name)
    model = model(pretrained=False)
    
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