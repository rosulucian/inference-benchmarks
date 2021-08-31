# inference-benchmarks

Scripts to measure performance rough performance gains for different deep learning inference engines

Models:
- pytorch vision models

Engines:
- PyTorch: CPU, cuda
- OnnxRuntime: CPU, cuda
- TorchScript (wip)
- OpenVINO (wip)


## Dependencies

* pytorch 1.9.0
* torchvision 0.10.0 
* OnnxRuntime 1.8.1 (optional)
* OpenVINO 2021.4 (optional)

## Usages
Native:
```
python benchmark/benchmark.py -f pytorch -m alexnet -b 5 -d cpu -e -v 
```
Docker:
```
# create directories
mkdir models results

# create onnx models files locally (otherwise will block container)
#

# build container
docker build -t pytbench -f docker/pytorch.Dockerfile .

# start container
docker run --rm -it \
-v ${PWD}/models:/app/models \
-v ${PWD}/results:/app/results \
pytbench

# run benchmark
python benchmark/benchmark.py -f pytorch -m alexnet -b 5 -d cpu -e -v
```

### Example

Check out some prerun benchmarks in this [notebook](notebooks/benchmark.ipynb)

## Notes

* Repository is work in progress
* Curently using python the python apis for the different engines; considering cpp apis as well
