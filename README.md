# CodeWithGPU

<p align="center">
  <img width="100%" src="https://dragon.seetatech.com/download/codewithgpu/assets/banner.png"/>
</>

[CodeWithGPU](https://www.codewithgpu.com) is a community that focuses on the reproducible AI algorithms. It has close links with [Github](https://www.github.com) by leveraging the managed code, and distributes corresponding docker images, models and logs for friendly reproduction.

This repository provides a novel data loading solution that maps data between Python object and serialized bytes automatically. This solution encourages developers to build a hierarchical data loading pipeline, which decouples the reading, transforming and batching. Similar solution, such as [NVIDIA DALI](https://developer.nvidia.com/dali), is widely deployed in many HPC systems and ML benchmarks.

Besides, it considers a modular and asynchronous design for the inference of AI models. Developers can easily serve their models on distributed devices by creating a many-to-many "Producer-Consumer" dataflow, and the flow control is dealt by the synchronous queues. By this way, model serving resembles training and can also get great benefit from the efficient data loader.

Also, it develops the benchmarks of modern AI models on diverse accelerators, including the newest NVIDIA GPUs and Apple Silicon processors. It will help users to match their demand on picking the best suitable devices. ***“The more reasonable GPUs you buy, the more money you save.”***

## Installation

Install from PyPI:

```bash
pip install codewithgpu
```

Or, clone this repository to local disk and install:

```bash
cd codewithgpu && pip instsall .
```

You can also install from the remote repository: 

```bash
pip install git+ssh://git@github.com/seetacloud/codewithgpu.git
```

## Quick Start

### Deploy Image Inference Application

See [Example: Image Inference](examples/image_inference.py).

### Use Record Dataset To Accelerate Data Loading

See [Example: Record Dataset](examples/record_dataset.py).

### Model Benchmarks

See [Doc: Model Benchmarks](benchmarks/models/README.md).

## License
[Apache License 2.0](LICENSE)
