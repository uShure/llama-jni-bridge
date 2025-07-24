#!/bin/bash

# Скрипт для запуска примера с правильными параметрами

if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_file> [prompt]"
    echo "Example: $0 tinyllama.gguf \"What is AI?\""
    exit 1
fi

MODEL_FILE=$1
PROMPT=${2:-"Once upon a time"}

# Проверяем файл модели
if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: Model file not found: $MODEL_FILE"
    echo "Run ./download_model.sh to download a model"
    exit 1
fi

# Определяем версию Java
JAVA_VERSION=$(java -version 2>&1 | head -n 1 | cut -d'"' -f 2 | cut -d'.' -f 1)

# Устанавливаем библиотечный путь для macOS/Linux
if [[ "$OSTYPE" == "darwin"* ]]; then
    export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$(pwd)/build/lib
else
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/build/lib
fi

# Компилируем если нужно
if [ ! -f "Example.class" ]; then
    echo "Compiling Example.java..."
    javac -cp tools/wrapper Example.java
fi

# Запускаем с правильными параметрами
if [ "$JAVA_VERSION" -ge 17 ]; then
    echo "Java $JAVA_VERSION detected, using --enable-native-access"
    java --enable-native-access=ALL-UNNAMED \
         -Djava.library.path=build/lib \
         -cp .:tools/wrapper \
         Example "$MODEL_FILE" "$PROMPT"
else
    echo "Java $JAVA_VERSION detected"
    java -Djava.library.path=build/lib \
         -cp .:tools/wrapper \
         Example "$MODEL_FILE" "$PROMPT"
fi
