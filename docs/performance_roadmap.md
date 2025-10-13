# GPU and High-Performance Processing Roadmap

## Overview
This roadmap tracks the work required to add accelerator-aware execution to
`yam_processor`. It summarises the candidate GPU backends under evaluation,
describes the interfaces modules must implement to opt into GPU execution, and
explains how OpenCV/scikit-image kernels will be wired into the existing
pipeline abstractions.

## Target GPU Backends
The initial focus is on backends with strong Python support and good alignment
with existing dependencies:

1. **CUDA (NVIDIA GPUs)** – Primary backend for desktop deployments using
   libraries such as CuPy and OpenCV's CUDA modules. CUDA acts as the reference
   implementation for API design and performance validation.
2. **ROCm (AMD GPUs)** – Evaluated via packages like ROCm-enabled PyTorch or
   CuPy-ROCm. Feature parity with CUDA is required before declaring the backend
   production ready.
3. **OpenCL/Vulkan Compute** – Longer-term option to support wider hardware
   coverage. Investigation will start after CUDA/ROCm land to avoid delaying the
   mainline acceleration path.
4. **Metal (Apple Silicon)** – Monitored as a stretch goal. Integration hinges on
   upstream availability of Metal-accelerated primitives in OpenCV or third-party
   bridges.

Each backend must expose a NumPy-compatible interface for array interchange
(e.g., DLPack) and provide fallbacks that gracefully return to CPU execution
without breaking pipeline semantics.

## Execution Metadata and Module Interfaces
GPU participation is coordinated via two primary entry points:

- `PipelineStep.execution.requires_gpu`: Steps that return `True` signal the
  pipeline manager to dispatch work to the configured `GpuExecutor`. The
  executor must accept the `PipelineStep` instance so it can access metadata and
  parameters while marshalling data to the accelerator.
- `ModuleCapabilities.requires_gpu`: Modules advertising GPU-only behaviour must
  ensure every generated `PipelineStep` mirrors the flag within its
  `StepExecutionMetadata`. Future validation hooks will compare module-level
  capabilities against per-step metadata to warn about mismatches.

Implementations must also populate `StepExecutionMetadata.supports_inplace` when
GPU kernels can operate on views or shared buffers, enabling zero-copy handoffs
between CPU and GPU domains.

## Integrating OpenCV and scikit-image Kernels
Planned integration will proceed in three phases:

1. **Kernel Cataloguing** – Identify high-value operations (denoising,
   morphological ops, edge detection) available in OpenCV CUDA modules and
   scikit-image. Define thin wrappers that translate module parameters into the
   appropriate function calls.
2. **Module Augmentation** – Extend existing preprocessing/segmentation modules
   to expose accelerator-aware variants of their kernels. Modules should reuse
   their current parameter schemas while switching the underlying implementation
   based on GPU availability.
3. **Pipeline Wiring** – Update `PipelineManager` to detect when a step can be
   executed with an optimised kernel, delegating to the `GpuExecutor` when a GPU
   backend is loaded, or falling back to CPU implementations otherwise.

Shared utilities (e.g., data conversion between NumPy arrays and device buffers)
will live alongside the GPU executor implementation so that future kernels can
opt in without repeating boilerplate.

## Milestones
- **M1 – Interface Stabilisation:** Finalise the `GpuExecutor` protocol,
  validation rules for `ModuleCapabilities`, and reference documentation.
- **M2 – CUDA Prototype:** Implement a proof-of-concept executor using CuPy and
  OpenCV CUDA functions, enabling one preprocessing and one segmentation module
  to run end-to-end on the GPU.
- **M3 – Backend Abstraction:** Introduce backend selection logic, environment
  probing, and comprehensive CPU fallbacks.
- **M4 – Extended Library Support:** Integrate scikit-image kernels and expand
  module coverage based on profiling feedback.

Progress should be tracked in project management tooling with cross-links to
pull requests that implement each milestone.
