import org.llm.wrapper.*;
import java.util.function.Predicate;

public class NewApiExample {
    public static void main(String[] args) {
        // Example 1: Chat mode (default)
        exampleChatMode();

        // Example 2: Plain text mode for better KV cache reuse
        examplePlainTextMode();
    }

    private static void exampleChatMode() {
        System.out.println("=== Chat Mode Example ===");

        // Initialize model
        ModelParams modelParams = new ModelParams();
        modelParams.model = "models/llama-2-7b.Q4_K_M.gguf";
        modelParams.ctxSize = 4096;
        modelParams.batchSize = 512;
        modelParams.gpuLayers = 35;  // Offload layers to GPU

        ComputeParams computeParams = new ComputeParams();
        computeParams.threads = 8;

        long handle = LlamaCpp.llama_init(modelParams, computeParams);
        if (handle == 0) {
            System.err.println("Failed to initialize: " + LlamaCpp.llama_get_error());
            return;
        }

        try {
            // Generate with streaming
            SamplingParams params = new SamplingParams();
            params.prompt = "Tell me a joke about programming";
            params.predict = 100;
            params.temp = 0.7f;
            params.topP = 0.9f;
            params.repeatPenalty = 1.1f;
            params.stream = true;
            params.chatMode = true;  // Use chat templates

            // Token callback for streaming
            Predicate<String> callback = token -> {
                System.out.print(token);
                System.out.flush();
                return true;  // Continue generation
            };

            int result = LlamaCpp.llama_generate(handle, params, callback);
            if (result != 0) {
                System.err.println("\nGeneration failed: " + LlamaCpp.llama_get_error());
            }
            System.out.println();

            // Second turn
            params.prompt = "Make it funnier!";
            result = LlamaCpp.llama_generate(handle, params, callback);
            System.out.println();

        } finally {
            LlamaCpp.llama_destroy(handle);
        }
    }

    private static void examplePlainTextMode() {
        System.out.println("\n=== Plain Text Mode Example ===");

        // Initialize model
        ModelParams modelParams = new ModelParams();
        modelParams.model = "models/llama-2-7b.Q4_K_M.gguf";
        modelParams.ctxSize = 4096;

        long handle = LlamaCpp.llama_init(modelParams, null);  // Use default compute params
        if (handle == 0) {
            System.err.println("Failed to initialize: " + LlamaCpp.llama_get_error());
            return;
        }

        try {
            // First generation
            SamplingParams params = new SamplingParams();
            params.prompt = "The capital of France is";
            params.predict = 20;
            params.temp = 0.1f;  // Low temperature for factual response
            params.chatMode = false;  // Plain text mode - no chat templates
            params.stream = false;

            System.out.print(params.prompt);
            int result = LlamaCpp.llama_generate(handle, params, token -> {
                System.out.print(token);
                return true;
            });

            // Continue with more text - KV cache will be reused
            params.prompt = "The capital of France is Paris. The capital of Germany is";
            System.out.print("\n\nContinuing with KV cache reuse:\n");
            System.out.print(params.prompt);

            result = LlamaCpp.llama_generate(handle, params, token -> {
                System.out.print(token);
                return true;
            });
            System.out.println();

        } finally {
            LlamaCpp.llama_destroy(handle);
        }
    }
}
