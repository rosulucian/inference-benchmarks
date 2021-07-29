# inference-benchmarks

Benchmarks for different deep learning inference engines

Models:

- pytorch vision models

Engines:

- pytorch: CPU, cuda
- onnxruntime: cpu, cuda, OpenVINO


## Dependencies

* pytorch 1.9.0
* torchvision 0.10.0
* OnnxRuntime 1.8.1
* OpenVINO 2021.4 [1^]

[1^]: To build OnnxRuntime with OpenVINO fillow instructions [here](https://www.onnxruntime.ai/docs/how-to/build/eps.html#openvino)

## Usages
```
python benchmark.py -f pytorch -m all -b 5 -d cpu cuda -e -v
```

### Example

Check out some prerun benchmarks in this [notebook](notebooks/benchmark.ipynb)
