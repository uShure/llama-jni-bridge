# Building llama-jni-bridge on macOS

## Prerequisites

1. Install Xcode Command Line Tools:
```bash
xcode-select --install
```

2. Install Homebrew (if not already installed):
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

3. Install CMake and Java:
```bash
brew install cmake openjdk@17
```

4. Set JAVA_HOME:
```bash
export JAVA_HOME=$(/usr/libexec/java_home)
# Add to your ~/.zshrc or ~/.bash_profile to make it permanent
```

## Building the Project

1. Clone the repository with submodules:
```bash
git clone https://github.com/uShure/llama-jni-bridge.git
cd llama-jni-bridge
git submodule update --init --recursive
```

2. Build using one of these options:

**Option A: Simplest build (recommended for quick start):**
```bash
./build_simple.sh
```
This builds only essential components without chat templates or json support.

**Option B: Full minimal build with dependencies:**
```bash
./build_minimal.sh
```

**Option C: CPU-only build if Metal causes issues:**
```bash
./build_cpu_only.sh
```

Or manually configure with cmake:
```bash
mkdir build
cd build
# Use the minimal CMakeLists for JNI only
cp ../CMakeLists-minimal.txt ../CMakeLists.txt
cmake ..
```

3. Build the project:
```bash
make -j$(sysctl -n hw.ncpu)
```

## Running the Example

1. Compile Java classes:
```bash
cd ../tools/wrapper
javac org/llm/wrapper/*.java
cd ../..
javac -cp tools/wrapper Example.java
```

2. Download a test model (example using Llama 2 7B):
```bash
# You can download any GGUF model, for example:
curl -L https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf -o llama-2-7b.Q4_K_M.gguf
```

3. Run the example:
```bash
java -Djava.library.path=build/tools/wrapper -cp .:tools/wrapper Example llama-2-7b.Q4_K_M.gguf "Once upon a time"
```

## Troubleshooting

### "Library not loaded" error on macOS
If you get a dyld error, you may need to set DYLD_LIBRARY_PATH:
```bash
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$(pwd)/build/tools/wrapper
```

### CMake can't find JNI
Make sure JAVA_HOME is set correctly:
```bash
echo $JAVA_HOME
# Should output something like: /Library/Java/JavaVirtualMachines/openjdk-17.jdk/Contents/Home
```

### Metal Performance Shaders
To enable GPU acceleration on Apple Silicon:
```bash
cmake -DGGML_METAL=ON ..
make -j$(sysctl -n hw.ncpu)
```

This will use Metal for GPU acceleration on M1/M2/M3 Macs.
