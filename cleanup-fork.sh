#!/bin/bash

# Cleanup script for llama-jni-bridge
# Removes unnecessary files from the llama.cpp fork

echo "Cleaning up llama-jni-bridge fork..."

# Remove example scripts that are not JNI-related
rm -f examples/*.sh examples/*.bat examples/*.py examples/*.vim
rm -f prompts/*.txt

# Remove Docker files
rm -rf .devops

# Remove CI/CD configurations
rm -rf .github/workflows
rm -f .github/labeler.yml
rm -f .pre-commit-config.yaml
rm -f .flake8
rm -f mypy.ini
rm -f pyrightconfig.json

# Remove Python-related files
rm -f convert_*.py
rm -f poetry.lock
rm -f pyproject.toml
rm -f requirements.txt
rm -rf requirements/
rm -rf gguf-py/

# Remove test models
rm -f models/*.gguf
rm -f models/*.inp
rm -f models/*.out

# Remove documentation not related to JNI
rm -f docs/android.md
rm -f docs/docker.md
rm -f docs/multimodal.md

# Remove scripts not needed for JNI
rm -f scripts/*.py
rm -f scripts/*.sh
rm -f scripts/*.bat
rm -f scripts/sync-ggml.last

# Remove example tools not related to JNI
rm -rf tools/batched-bench
rm -rf tools/cvector-generator
rm -rf tools/export-lora
rm -rf tools/gguf-split
rm -rf tools/imatrix
rm -rf tools/llama-bench
rm -rf tools/main
rm -rf tools/perplexity
rm -rf tools/quantize
rm -rf tools/rpc
rm -rf tools/run
rm -rf tools/server
rm -rf tools/tokenize
rm -rf tools/tts
rm -rf tools/mtmd

# Remove test files not related to JNI
rm -f tests/*.py
rm -f tests/*.sh
rm -f tests/*.mjs

# Clean up build artifacts
rm -rf build/
rm -rf build-manual/
rm -f llama.h.new

echo "Cleanup complete!"
echo ""
echo "Remaining structure:"
echo "- Core llama.cpp source files (src/, include/, ggml/)"
echo "- Common utilities (common/)"
echo "- JNI wrapper (tools/wrapper/)"
echo "- Build configuration (CMakeLists.txt files)"
echo "- Documentation (README.md, README-JNI.md, LICENSE)"
echo "- Example Java application (Example.java)"
