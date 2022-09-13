# Model Benchmarks

## Quick Start

```
cd codewithgpu/benchmarks
python bench_model.py -f ./results.json
```

For more usages, see "--help" argument:

```
python bench_model.py --help
```

## Training Baselines

### ResNet50

| Backend | Device | Prec | Perf (TFLOPS) | Time (FPS) |
| :-----: | :----: | :--: | :-----------: | :--------: |
| torch_vm | M1 Pro | FP32 | 4.6 | 61 |
| torch_vm | TITAN V | FP16 | 110 | 661 |
| torch_vm | TITAN V | FP32 | 14.9 | 289 |

### ViT-Base

| Backend | Device | Prec | Perf (TFLOPS) | Time (FPS) |
| :-----: | :----: | :--: | :-----------: | :--------: |
| torch_vm | M1 Pro | FP32 | 4.6 | 22 |
| torch_vm | TITAN V | FP16 | 110 | 333 |
| torch_vm | TITAN V | FP32 | 14.9 | 86 |

### MobileNetV3

| Backend | Device | Prec | Perf (TFLOPS) | Time (FPS) |
| :-----: | :----: | :--: | :-----------: | :--------: |
| torch_vm | M1 Pro | FP32 | 4.6 | 85 |
| torch_vm | TITAN V | FP16 | 110 | 1527 |
| torch_vm | TITAN V | FP32 | 14.9 | 878 |

## Inference Baselines

### ResNet50

| Backend | Device | Prec | Perf (TFLOPS) | Time (FPS) |
| :-----: | :----: | :--: | :-----------: | :--------: |
| torch_vm | M1 Pro | FP32 | 4.6 | 214 |
| torch_vm | TITAN V | FP16 | 110 | 2071 |
| torch_vm | TITAN V | FP32 | 14.9 | 940 |

### ViT-Base

| Backend | Device | Prec | Perf (TFLOPS) | Time (FPS) |
| :-----: | :----: | :--: | :-----------: | :--------: |
| torch_vm | M1 Pro | FP32 | 4.6 | 61 |
| torch_vm | TITAN V | FP16 | 110 | 1033 |
| torch_vm | TITAN V | FP32 | 14.9 | 262 |

### MobileNetV3

| Backend | Device | Prec | Perf (TFLOPS) | Time (FPS) |
| :-----: | :----: | :--: | :-----------: | :--------: |
| torch_vm | M1 Pro | FP32 | 4.6 | 382 |
| torch_vm | TITAN V | FP16 | 110 | 6504 |
| torch_vm | TITAN V | FP32 | 14.9 | 3807 |
