package org.llm.wrapper;

import java.util.function.Predicate;

/**
 * Improved API design for LlamaCpp JNI wrapper
 * This version moves the token callback out of GenerateParams
 */
public class LlamaCppImproved {
    static {
        System.loadLibrary("java_wrapper");
    }

    // Native methods matching the original API
    public static native long llama_init(InitParams params);
    public static native void llama_destroy(long handle);
    public static native String llama_get_error();
    public static native long llama_get_native_memory_usage(long handle);
    public static native long llama_get_os_memory_usage();
    public static native void llama_clear_chat(long handle);

    // Original generate method for backward compatibility
    public static native int llama_generate(long handle, GenerateParams params);

    // Improved generate method with separate callback parameter
    public static int generate(long handle, GenerateParams params, Predicate<String> tokenCallback) {
        // Store the callback temporarily in params for the native call
        Predicate<String> oldCallback = params.tokenCallback;
        params.tokenCallback = tokenCallback;

        try {
            return llama_generate(handle, params);
        } finally {
            // Restore original callback
            params.tokenCallback = oldCallback;
        }
    }

    // Convenience method without callback
    public static int generate(long handle, GenerateParams params) {
        return generate(handle, params, null);
    }

    // Convenience method to generate and return complete response
    public static String generateString(long handle, GenerateParams params) {
        StringBuilder response = new StringBuilder();

        int result = generate(handle, params, token -> {
            response.append(token);
            return true; // Continue generation
        });

        if (result != 0) {
            throw new RuntimeException("Generation failed: " + llama_get_error());
        }

        return response.toString();
    }

    // Builder pattern for cleaner parameter construction
    public static class InitParamsBuilder {
        private InitParams params = new InitParams();

        public InitParamsBuilder modelPath(String path) {
            params.modelPath = path;
            return this;
        }

        public InitParamsBuilder contextSize(int size) {
            params.nCtx = size;
            return this;
        }

        public InitParamsBuilder batchSize(int size) {
            params.nBatch = size;
            return this;
        }

        public InitParamsBuilder threads(int threads) {
            params.nThreads = threads;
            return this;
        }

        public InitParamsBuilder gpuLayers(int layers) {
            params.nGpuLayers = layers;
            return this;
        }

        public InitParamsBuilder seed(int seed) {
            params.seed = seed;
            return this;
        }

        public InitParamsBuilder keepTokens(int n) {
            params.nKeep = n;
            return this;
        }

        public InitParams build() {
            return params;
        }
    }

    public static class GenerateParamsBuilder {
        private GenerateParams params = new GenerateParams();

        public GenerateParamsBuilder prompt(String prompt) {
            params.prompt = prompt;
            return this;
        }

        public GenerateParamsBuilder maxTokens(int n) {
            params.nPredict = n;
            return this;
        }

        public GenerateParamsBuilder temperature(float temp) {
            params.temp = temp;
            return this;
        }

        public GenerateParamsBuilder topK(int k) {
            params.topK = k;
            return this;
        }

        public GenerateParamsBuilder topP(float p) {
            params.topP = p;
            return this;
        }

        public GenerateParamsBuilder repeatPenalty(float penalty) {
            params.repeatPenalty = penalty;
            return this;
        }

        public GenerateParamsBuilder seed(int seed) {
            params.seed = seed;
            return this;
        }

        public GenerateParamsBuilder plainTextMode() {
            params.chatMode = false;
            return this;
        }

        public GenerateParamsBuilder chatMode() {
            params.chatMode = true;
            return this;
        }

        public GenerateParams build() {
            return params;
        }
    }

    // Example usage
    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: java LlamaCppImproved <model_path>");
            return;
        }

        // Initialize with builder pattern
        InitParams initParams = new InitParamsBuilder()
            .modelPath(args[0])
            .contextSize(2048)
            .threads(4)
            .gpuLayers(0)
            .seed(42)
            .keepTokens(128)
            .build();

        long handle = llama_init(initParams);
        if (handle == 0) {
            System.err.println("Failed to initialize: " + llama_get_error());
            return;
        }

        try {
            // Example 1: Generate with streaming callback
            GenerateParams genParams = new GenerateParamsBuilder()
                .prompt("The capital of France is")
                .maxTokens(50)
                .temperature(0.7f)
                .plainTextMode()
                .build();

            System.out.println("Streaming generation:");
            int result = generate(handle, genParams, token -> {
                System.out.print(token);
                System.out.flush();
                return true;
            });
            System.out.println();

            // Example 2: Generate complete string
            genParams = new GenerateParamsBuilder()
                .prompt("What is the meaning of life?")
                .maxTokens(100)
                .temperature(0.8f)
                .plainTextMode()
                .build();

            System.out.println("\nComplete generation:");
            String response = generateString(handle, genParams);
            System.out.println(response);

        } finally {
            llama_destroy(handle);
        }
    }
}
