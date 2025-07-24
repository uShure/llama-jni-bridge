# Quick Start Guide for llama-jni-bridge

This is the fastest way to get llama-jni-bridge running on your system.

## Prerequisites
- Git
- C++ compiler (Xcode on macOS, GCC on Linux)
- Java JDK 8 or higher
- CMake 3.14 or higher

## Step 1: Clone and Build

```bash
# Clone the repository
git clone https://github.com/uShure/llama-jni-bridge.git
cd llama-jni-bridge

# Initialize submodules
git submodule update --init --recursive

# Run the simple build script
./build_simple.sh
```

## Step 2: Compile Java Classes

```bash
# Compile the wrapper classes
cd tools/wrapper
javac org/llm/wrapper/*.java

# Go back to project root
cd ../..

# Compile the example
javac -cp tools/wrapper Example.java
```

## Step 3: Download a Test Model

```bash
# Download a small test model (TinyLlama 1.1B)
curl -L https://huggingface.co/TheBloke/TinyLlama-1.1B-GGUF/resolve/main/tinyllama-1.1b.Q4_K_M.gguf -o tinyllama.gguf
```

## Step 4: Run the Example

```bash
java -Djava.library.path=build/lib -cp .:tools/wrapper Example tinyllama.gguf "Once upon a time"
```

## Expected Output

You should see:
1. "Loading model from: tinyllama.gguf"
2. "Model loaded successfully!"
3. Generated text streaming to console
4. "Generation completed successfully!"

## Troubleshooting

### "UnsatisfiedLinkError"
Make sure the library path is correct:
```bash
ls build/lib/  # Should show libjava_wrapper.dylib (macOS) or .so (Linux)
```

### Build fails with undefined symbols
The build scripts now automatically generate the required build-info.cpp file. If you still have issues:
```bash
rm -rf build
rm -f common/build-info.cpp
./build_simple.sh
```

### Other build failures
Try cleaning and rebuilding:
```bash
rm -rf build
./build_simple.sh
```

### Model not found
Make sure you downloaded the model to the correct location.

## Next Steps

- Try different models from [Hugging Face](https://huggingface.co/models?library=gguf)
- Modify `Example.java` to experiment with different parameters
- Integrate into your own Java application

## Using in Your Project

```java
// Minimal usage example
LlamaCpp llama = new LlamaCpp();

InitParams params = new InitParams();
params.modelPath = "path/to/model.gguf";
params.nCtx = 2048;

long handle = llama.llama_init(params);
if (handle != 0) {
    GenerateParams genParams = new GenerateParams();
    genParams.prompt = "Hello world";
    genParams.nPredict = 50;

    llama.llama_generate(handle, genParams);
    llama.llama_exit(handle);
}
```
