# 📜 Available Scripts Guide

## 🚀 Repository Management

### `push-to-github.sh`
Push your repository to GitHub with proper authentication handling.
```bash
./push-to-github.sh
```

## 🔨 Build Scripts

### `compile-java.sh`
Compile the Java wrapper classes.
```bash
./compile-java.sh
```

## 📚 Java Examples

### `ExampleUsage.java`
Interactive chat example with the LLM.
```bash
javac -cp . ExampleUsage.java
java -Djava.library.path=./build/tools/wrapper ExampleUsage model.gguf
```

### `BatchExample.java`
Batch generation example for multiple prompts.
```bash
javac -cp . BatchExample.java
java -Djava.library.path=./build/tools/wrapper BatchExample model.gguf
```

### `Example.java` (Original)
Basic example showing simple generation.
```bash
javac -cp . Example.java
java -Djava.library.path=./build/tools/wrapper Example
```

## 🧪 Testing

### C++ Test Program
Test the wrapper directly without Java:
```bash
./build/bin/wrapper_test model.gguf "Your prompt"
```

## 📋 Quick Start Sequence

1. **Build the project:**
   ```bash
   cmake -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build -j 8
   ```

2. **Compile Java classes:**
   ```bash
   ./compile-java.sh
   ```

3. **Test with C++:**
   ```bash
   ./build/bin/wrapper_test model.gguf
   ```

4. **Test with Java:**
   ```bash
   javac -cp . ExampleUsage.java
   java -Djava.library.path=./build/tools/wrapper ExampleUsage model.gguf
   ```

5. **Push to GitHub:**
   ```bash
   ./push-to-github.sh
   ```

## 💡 Tips

- Always set `java.library.path` to point to the JNI library location
- Use smaller quantized models (Q4_K_M) for faster testing
- Adjust thread count based on your CPU cores
- For Apple Silicon, add `-DGGML_METAL=ON` to CMake for GPU support

## 🔍 Troubleshooting

### Can't find JNI library:
```bash
export LD_LIBRARY_PATH=$PWD/build/tools/wrapper:$LD_LIBRARY_PATH
```

### Java class not found:
```bash
export CLASSPATH=$PWD:$PWD/tools/wrapper:$CLASSPATH
```

### On macOS:
```bash
export DYLD_LIBRARY_PATH=$PWD/build/tools/wrapper:$DYLD_LIBRARY_PATH
```
