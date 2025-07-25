#!/bin/bash

# Скрипт для автоматического тестирования llama-jni-bridge

set -e

echo "=== LLaMA JNI Bridge Test Suite ==="
echo

# Проверка наличия модели
if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_path>"
    echo "Example: $0 tinyllama.gguf"
    exit 1
fi

MODEL_PATH=$1

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH"
    exit 1
fi

# Определение платформы
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/build/lib
elif [[ "$OSTYPE" == "darwin"* ]]; then
    export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$(pwd)/build/lib
fi

# Компиляция тестов
echo "Compiling tests..."
javac -cp tools/wrapper Example.java
javac -cp tools/wrapper SimpleTest.java
javac -cp tools/wrapper TestLlama.java
javac -cp tools/wrapper BenchmarkLlama.java

echo "Compilation complete!"
echo

# Тест 1: Базовый пример
echo "=== Test 1: Basic Example ==="
java -Djava.library.path=build/lib -cp .:tools/wrapper Example "$MODEL_PATH" "Once upon a time" | head -20
echo
echo "--- Test 1 Complete ---"
echo

# Тест 2: Простой тест
echo "=== Test 2: Simple Test ==="
java -Djava.library.path=build/lib -cp .:tools/wrapper SimpleTest "$MODEL_PATH" "The meaning of life is"
echo
echo "--- Test 2 Complete ---"
echo

# Тест 3: Проверка памяти
echo "=== Test 3: Memory Test ==="
echo "Initial memory:"
ps aux | grep java | grep -v grep || echo "No Java processes"

echo "Running generation..."
java -Djava.library.path=build/lib -cp .:tools/wrapper SimpleTest "$MODEL_PATH" "Write a short story about" > /dev/null 2>&1

echo "After generation:"
ps aux | grep java | grep -v grep || echo "No Java processes"
echo "--- Test 3 Complete ---"
echo

# Тест 4: Проверка утечек памяти (многократный запуск)
echo "=== Test 4: Memory Leak Test ==="
for i in {1..5}; do
    echo -n "Run $i... "
    java -Djava.library.path=build/lib -cp .:tools/wrapper SimpleTest "$MODEL_PATH" "Test $i" > /dev/null 2>&1
    echo "done"
done
echo "--- Test 4 Complete ---"
echo

echo "=== All tests completed successfully! ==="
echo
echo "To run interactive test: java -Djava.library.path=build/lib -cp .:tools/wrapper TestLlama $MODEL_PATH"
echo "To run benchmark: java -Djava.library.path=build/lib -cp .:tools/wrapper BenchmarkLlama $MODEL_PATH"
