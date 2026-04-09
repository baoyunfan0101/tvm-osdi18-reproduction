# TVM OSDI '18 Reproduction

## Environment

- Python: 3.9–3.11
- NumPy: < 2.0
- TVM
  - Version: `0.14.dev273` (installed from pip) / `0.15.dev273` (built from source)
  - Commit hash: `878a61105ea4c85f3547fe137a28d0a80b1f0e94`

## Setup

### Colab

Runtime -> Change runtime type -> Runtime version -> 2025.07

```bash
bash env/colab_setup.sh
```

### macOS

```bash
bash env/mac_setup.sh
```

## Experiment 1: Graph Level

### Run

```bash
# macOS
python -m exp1_graph_level.run_fusion --target llvm --platform mac_cpu --workload conv_bias_relu
python -m exp1_graph_level.run_fusion --target llvm --platform mac_cpu --workload conv_bn_relu

# Google Colab
python -m exp1_graph_level.run_fusion --target llvm --platform colab_cpu --workload conv_bias_relu
python -m exp1_graph_level.run_fusion --target llvm --platform colab_cpu --workload conv_bn_relu

!python -m exp1_graph_level.run_fusion --target cuda --platform colab_gpu --workload conv_bias_relu
!python -m exp1_graph_level.run_fusion --target cuda --platform colab_gpu --workload conv_bn_relu
```

### Plot

```bash
python -m exp1_graph_level.plot
```

## Experiment 2: Operator Level

### Run

```bash
# macOS
python -m exp2_operator_level.run_tuning --target llvm --platform mac_cpu --workload conv2d
python -m exp2_operator_level.run_tuning --target llvm --platform mac_cpu --workload depthwise

# Google Colab
!python -m exp2_operator_level.run_tuning --target llvm --platform colab_cpu --workload conv2d
!python -m exp2_operator_level.run_tuning --target llvm --platform colab_cpu --workload depthwise

!python -m exp2_operator_level.run_tuning --target cuda --platform colab_gpu --workload conv2d
!python -m exp2_operator_level.run_tuning --target cuda --platform colab_gpu --workload depthwise
```

### Plot

```bash
python -m exp2_operator_level.plot
```

## Experiment 3: End-to-End

### Run on CPU

```bash
# macOS
python -m exp3_end2end.run_end2end --target llvm --platform mac_cpu --model resnet
python -m exp3_end2end.run_end2end --target llvm --platform mac_cpu --model mobilenet

# Google Colab
!python -m exp3_end2end.run_end2end --target llvm --platform colab_cpu --model resnet
!python -m exp3_end2end.run_end2end --target llvm --platform colab_cpu --model mobilenet

!python -m exp3_end2end.run_end2end --target cuda --platform colab_gpu --model resnet
!python -m exp3_end2end.run_end2end --target cuda --platform colab_gpu --model mobilenet
```

### Plot

```bash
python -m exp3_end2end.plot
```