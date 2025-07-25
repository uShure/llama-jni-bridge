# API Changes for llama-jni-bridge

## Overview

This update introduces a new, more flexible API for the llama-jni-bridge while maintaining backward compatibility through deprecated wrapper methods.

## New Parameter Classes

### ModelParams
Parameters for model loading and context initialization:
- `model` - Path to the model file
- `ctxSize` - Context size (default: 4096)
- `batchSize` - Batch size (default: 2048)
- `gpuLayers` - Number of layers to offload to GPU
- `flashAttn` - Enable Flash Attention
- `cacheTypeK/V` - KV cache data types
- And many more...

### ComputeParams
CPU/GPU computation parameters:
- `threads` - Number of threads for generation
- `threadsBatch` - Number of threads for batch processing
- `cpuMask` - CPU affinity mask
- `priority` - Process priority
- etc.

### SamplingParams
Text generation parameters:
- `prompt` - Input prompt
- `predict` - Number of tokens to generate (-1 for infinite)
- `temp`, `topK`, `topP`, `minP` - Sampling parameters
- `repeatPenalty`, `presencePenalty`, `frequencyPenalty` - Penalties
- `chatMode` - Enable chat templates (default: true)
- `stream` - Enable streaming output

## New API Methods

```java
// Initialize with separate parameter objects
long llama_init(ModelParams modelParams, ComputeParams computeParams);

// Generate with callback passed separately
int llama_generate(long handle, SamplingParams params, Predicate<String> tokenCallback);
```

## Key Features

### 1. Plain Text Mode
Set `chatMode = false` in SamplingParams to use plain text mode without chat templates. This enables better KV cache reuse for continuation-style generation.

### 2. Improved Sampling
Support for all llama.cpp sampling parameters including:
- Min-p sampling
- Typical sampling
- Presence and frequency penalties
- DRY sampling
- Mirostat

### 3. Better Parameter Names
Parameters now match llama.cpp command-line arguments (e.g., `predict` instead of `nPredict`).

## Backward Compatibility

The old API is still supported through deprecated methods:

```java
@Deprecated
long llama_init(InitParams params);

@Deprecated
int llama_generate(long handle, GenerateParams params);
```

## Migration Guide

### Old API:
```java
InitParams init = new InitParams();
init.modelPath = "model.gguf";
init.nCtx = 2048;
init.nThreads = 8;

GenerateParams gen = new GenerateParams();
gen.prompt = "Hello";
gen.nPredict = 100;
gen.tokenCallback = token -> { print(token); return true; };

long handle = LlamaCpp.llama_init(init);
LlamaCpp.llama_generate(handle, gen);
```

### New API:
```java
ModelParams model = new ModelParams();
model.model = "model.gguf";
model.ctxSize = 2048;

ComputeParams compute = new ComputeParams();
compute.threads = 8;

SamplingParams sampling = new SamplingParams();
sampling.prompt = "Hello";
sampling.predict = 100;

long handle = LlamaCpp.llama_init(model, compute);
LlamaCpp.llama_generate(handle, sampling, token -> { print(token); return true; });
```

## Parameter Mapping

See `.same/parameter-mapping.md` for a complete mapping of string parameter names to Java fields and context fields.
