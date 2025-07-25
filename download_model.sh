#!/bin/bash

# Скрипт для загрузки и проверки моделей

echo "=== Model Downloader for llama-jni-bridge ==="
echo

# Функция для загрузки с прогрессом
download_with_progress() {
    local url=$1
    local output=$2
    echo "Downloading: $output"
    curl -L --progress-bar "$url" -o "$output"
}

# Функция для проверки GGUF файла
check_gguf_file() {
    local file=$1
    if [ ! -f "$file" ]; then
        echo "Error: File not found: $file"
        return 1
    fi

    # Проверяем magic bytes (должно быть GGUF)
    magic=$(head -c 4 "$file" 2>/dev/null)
    if [ "$magic" = "GGUF" ]; then
        echo "✓ Valid GGUF file: $file"
        # Показываем размер файла
        ls -lh "$file" | awk '{print "  Size: " $5}'
        return 0
    else
        echo "✗ Invalid GGUF file: $file (magic: '$magic')"
        return 1
    fi
}

# Меню выбора модели
echo "Select a model to download:"
echo "1) TinyLlama 1.1B (Q4_K_M) - 669 MB"
echo "2) TinyLlama 1.1B (Q8_0) - 1.17 GB"
echo "3) Phi-3 Mini 4K (Q4) - 2.3 GB"
echo "4) Llama 2 7B (Q4_K_M) - 3.8 GB"
echo "5) Custom URL"
echo "0) Check existing models"
echo

read -p "Enter your choice (0-5): " choice

case $choice in
    1)
        MODEL_URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-GGUF/resolve/main/tinyllama-1.1b.Q4_K_M.gguf"
        MODEL_FILE="tinyllama-1.1b-q4.gguf"
        ;;
    2)
        MODEL_URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-GGUF/resolve/main/tinyllama-1.1b.Q8_0.gguf"
        MODEL_FILE="tinyllama-1.1b-q8.gguf"
        ;;
    3)
        MODEL_URL="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf"
        MODEL_FILE="phi3-mini-q4.gguf"
        ;;
    4)
        MODEL_URL="https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf"
        MODEL_FILE="llama2-7b-q4.gguf"
        ;;
    5)
        read -p "Enter model URL: " MODEL_URL
        read -p "Enter output filename: " MODEL_FILE
        ;;
    0)
        echo "Checking existing GGUF files..."
        for file in *.gguf; do
            if [ -f "$file" ]; then
                check_gguf_file "$file"
            fi
        done
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

# Проверяем, существует ли файл
if [ -f "$MODEL_FILE" ]; then
    echo "File already exists: $MODEL_FILE"
    read -p "Overwrite? (y/n): " overwrite
    if [ "$overwrite" != "y" ]; then
        echo "Checking existing file..."
        check_gguf_file "$MODEL_FILE"
        exit 0
    fi
fi

# Загружаем модель
download_with_progress "$MODEL_URL" "$MODEL_FILE"

# Проверяем загруженный файл
echo
echo "Verifying downloaded file..."
if check_gguf_file "$MODEL_FILE"; then
    echo
    echo "Model downloaded successfully!"
    echo
    echo "To test the model, run:"
    echo "java --enable-native-access=ALL-UNNAMED -Djava.library.path=build/lib -cp .:tools/wrapper Example $MODEL_FILE \"What is the meaning of life?\""
else
    echo
    echo "Download failed or file is corrupted. Removing..."
    rm -f "$MODEL_FILE"
    exit 1
fi
