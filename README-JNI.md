# LLaMA JNI Bridge

A Java Native Interface (JNI) wrapper for llama.cpp, allowing Java applications to use LLaMA models for text generation.

**Quick Start:** See [QUICKSTART.md](QUICKSTART.md) for the fastest way to get running.

## Features

- Simple Java API for LLaMA model loading and text generation
- Streaming token generation with callback support
- Thread-safe model and context management
- Support for GPU acceleration via CUDA/Metal
- Configurable generation parameters (temperature, top-k, top-p, etc.)

## Prerequisites

- CMake 3.13 or higher
- C++ compiler with C++17 support
- Java Development Kit (JDK) 8 or higher
- CUDA toolkit (optional, for GPU support)
- Internet connection (to download nlohmann/json and minja dependencies)

## Building

### 1. Clone the repository

```bash
git clone https://github.com/uShure/llama-jni-bridge.git
cd llama-jni-bridge

# Important: Initialize the ggml submodule
git submodule update --init --recursive
```

### 2. Build the native library

```bash
mkdir build
cd build
cmake ..
make
```

For GPU support with CUDA:
```bash
cmake -DGGML_CUDA=ON ..
make
```

### 3. Build the Java classes

```bash
cd ../tools/wrapper
javac org/llm/wrapper/*.java
```

## Usage

### 1. Set up the library path

Make sure the compiled native library is in your system's library path:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/llama-jni-bridge/build/tools/wrapper
```

On macOS:
```bash
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/path/to/llama-jni-bridge/build/tools/wrapper
```

### 2. Example Java code

```java
import org.llm.wrapper.*;
import java.util.function.Predicate;

public class Example {
    public static void main(String[] args) {
        LlamaCpp llama = new LlamaCpp();

        // Initialize model
        InitParams initParams = new InitParams();
        initParams.modelPath = "path/to/your/model.gguf";
        initParams.nCtx = 2048;        // Context size
        initParams.nGpuLayers = 35;   // Number of layers to offload to GPU
        initParams.nThreads = 8;       // Number of CPU threads

        long handle = llama.llama_init(initParams);
        if (handle == 0) {
            System.err.println("Failed to initialize model");
            return;
        }

        // Generate text
        GenerateParams genParams = new GenerateParams();
        genParams.prompt = "Once upon a time";
        genParams.nPredict = 100;      // Max tokens to generate
        genParams.temp = 0.8f;         // Temperature
        genParams.topK = 40;           // Top-K sampling
        genParams.topP = 0.95f;        // Top-P sampling
        genParams.stream = true;       // Enable streaming

        // Set up streaming callback
        genParams.tokenCallback = new Predicate<String>() {
            @Override
            public boolean test(String token) {
                System.out.print(token);
                return true; // Return false to stop generation
            }
        };

        int result = llama.llama_generate(handle, genParams);

        // Clean up
        llama.llama_destroy(handle);
    }
}
```

### 3. Run the example

```bash
java -Djava.library.path=/path/to/llama-jni-bridge/build/tools/wrapper -cp /path/to/tools/wrapper Example
```

## API Reference

### InitParams

- `modelPath` (String): Path to the GGUF model file
- `nCtx` (int): Context size (default: 512)
- `nBatch` (int): Batch size for prompt processing (default: 512)
- `nThreads` (int): Number of CPU threads (default: 4)
- `nThreadsBatch` (int): Number of threads for batch processing (default: 4)
- `nGpuLayers` (int): Number of layers to offload to GPU (default: 0)
- `seed` (int): Random seed, -1 for random (default: -1)
- `useMmap` (boolean): Use memory mapping for model (default: true)
- `useMlock` (boolean): Lock model in memory (default: false)
- `embeddings` (boolean): Enable embeddings mode (default: false)

### GenerateParams

- `prompt` (String): Input prompt text
- `nPredict` (int): Maximum tokens to generate (default: 512)
- `nThreads` (int): Number of threads for generation (default: 4)
- `nBatch` (int): Batch size (default: 512)
- `topK` (int): Top-K sampling parameter (default: 40)
- `topP` (float): Top-P sampling parameter (default: 0.95)
- `temp` (float): Temperature for sampling (default: 0.8)
- `repeatPenalty` (float): Repetition penalty (default: 1.1)
- `repeatLastN` (int): Number of tokens to consider for repetition (default: 64)
- `seed` (int): Random seed, -1 for random (default: -1)
- `stream` (boolean): Enable streaming mode (default: true)
- `tokenCallback` (Predicate<String>): Callback for streaming tokens

## Troubleshooting

### "UnsatisfiedLinkError: no java_wrapper in java.library.path"

Make sure the native library is built and the library path is set correctly.

### "Failed to load model"

- Check that the model file exists and is a valid GGUF file
- Ensure you have enough RAM/VRAM for the model
- Try reducing `nGpuLayers` if you're running out of VRAM

### Compilation errors

Make sure you have:
- CMake 3.13+
- C++ compiler with C++17 support
- JDK installed with `javac` in PATH

## Common Build Issues

### Metal Configuration Error on macOS
If you get "Read-only file system" errors when building with Metal:
```bash
# Use the CPU-only build instead:
./build_cpu_only.sh
```

### Alternative: Fix Metal permissions
```bash
# Create build directory with proper permissions
mkdir -p build/bin
chmod -R 755 build
cd build
cmake -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release ..
make
```

## License

This project inherits the MIT license from llama.cpp.
