#!/bin/bash
# Script to compile Java wrapper classes

echo "☕ Compiling Java wrapper classes..."
echo ""

cd tools/wrapper

# Compile all Java files
echo "Compiling org.llm.wrapper classes..."
javac org/llm/wrapper/*.java

if [ $? -eq 0 ]; then
    echo "✅ Java classes compiled successfully!"
    echo ""
    echo "Compiled classes:"
    find org -name "*.class" -type f
else
    echo "❌ Java compilation failed!"
    exit 1
fi

echo ""
echo "To use these classes:"
echo "1. Include them in your Java classpath"
echo "2. Load the JNI library: System.loadLibrary(\"java_wrapper\")"
echo "3. Use LlamaCpp.llama_init() to start"
