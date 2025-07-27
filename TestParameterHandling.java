import org.llm.wrapper.LlamaCpp;
import org.llm.wrapper.InitParams;
import org.llm.wrapper.GenerateParams;
import java.util.function.Predicate;

public class TestParameterHandling {
    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: java TestParameterHandling <model_path>");
            System.exit(1);
        }

        String modelPath = args[0];
        System.out.println("=== Java Parameter Handling Test ===");
        System.out.println("Model: " + modelPath);

        // Test 1: Initialize with custom parameters
        System.out.println("\n1. Testing initialization with custom parameters...");
        InitParams initParams = new InitParams();
        initParams.modelPath = modelPath;
        initParams.nCtx = 2048;
        initParams.nBatch = 512;
        initParams.nThreads = 4;
        initParams.seed = 42; // Custom seed
        initParams.nGpuLayers = 0;
        initParams.useMmap = true;
        initParams.useMlock = false;
        initParams.cacheTypeK = 1; // f16
        initParams.cacheTypeV = 1; // f16
        initParams.nKeep = 128; // Keep first 128 tokens for caching

        long handle = LlamaCpp.llama_init(initParams);
        if (handle == 0) {
            System.err.println("Failed to initialize: " + LlamaCpp.llama_get_error());
            System.exit(1);
        }
        System.out.println("Initialized successfully with handle: " + handle);

        // Test 2: Generate with all parameters
        System.out.println("\n2. Testing generation with all parameters...");
        GenerateParams genParams = new GenerateParams();
        genParams.prompt = "The capital of France is";
        genParams.nPredict = 50;
        genParams.temp = 0.7f;
        genParams.topK = 40;
        genParams.topP = 0.95f;
        genParams.minP = 0.05f;
        genParams.repeatPenalty = 1.1f;
        genParams.repeatLastN = 64;
        genParams.seed = 42; // Same seed for reproducibility
        genParams.chatMode = false; // Plain text mode - no templates

        // Test token callback
        final StringBuilder response = new StringBuilder();
        Predicate<String> tokenCallback = token -> {
            System.out.print(token);
            response.append(token);
            return true; // Continue generation
        };
        genParams.tokenCallback = tokenCallback;

        int result = LlamaCpp.llama_generate(handle, genParams);
        if (result != 0) {
            System.err.println("\nGeneration failed: " + LlamaCpp.llama_get_error());
        } else {
            System.out.println("\nGeneration completed successfully");
        }

        // Test 3: Test caching with prefix reuse
        System.out.println("\n\n3. Testing caching with prefix reuse...");

        // First, generate with a longer prompt that extends the previous one
        genParams.prompt = "The capital of France is Paris. What about Germany?";
        genParams.nPredict = 30;

        System.out.println("Prompt 2: " + genParams.prompt);
        System.out.print("Response 2: ");

        result = LlamaCpp.llama_generate(handle, genParams);
        if (result != 0) {
            System.err.println("\nSecond generation failed: " + LlamaCpp.llama_get_error());
        } else {
            System.out.println("\nSecond generation completed - should have reused cache");
        }

        // Test 4: Test with different sampling parameters
        System.out.println("\n\n4. Testing different sampling parameters...");

        // Mirostat v2
        genParams.prompt = "Once upon a time";
        genParams.mirostat = 2;
        genParams.mirostatTau = 5.0f;
        genParams.mirostatEta = 0.1f;
        genParams.temp = 0.0f; // Mirostat controls temperature
        genParams.nPredict = 50;

        System.out.println("Testing Mirostat v2 sampling...");
        System.out.print("Response: ");

        result = LlamaCpp.llama_generate(handle, genParams);
        if (result != 0) {
            System.err.println("\nMirostat generation failed: " + LlamaCpp.llama_get_error());
        }

        // DRY sampling
        genParams.mirostat = 0; // Disable mirostat
        genParams.temp = 0.8f;
        genParams.dryMultiplier = 0.8f;
        genParams.dryBase = 1.75f;
        genParams.dryAllowedLength = 2;
        genParams.dryPenaltyLastN = 256;

        System.out.println("\n\nTesting DRY sampling...");
        genParams.prompt = "The quick brown fox";
        System.out.print("Response: ");

        result = LlamaCpp.llama_generate(handle, genParams);
        if (result != 0) {
            System.err.println("\nDRY generation failed: " + LlamaCpp.llama_get_error());
        }

        // Test 5: Memory usage
        System.out.println("\n\n5. Testing memory measurement...");
        long nativeMemory = LlamaCpp.llama_get_native_memory_usage(handle);
        long osMemory = LlamaCpp.llama_get_os_memory_usage();

        System.out.println("Native memory usage: " + (nativeMemory / (1024.0 * 1024.0)) + " MB");
        System.out.println("OS-reported memory: " + (osMemory / (1024.0 * 1024.0)) + " MB");

        // Test 6: Chat mode vs plain text mode
        System.out.println("\n\n6. Testing chat mode vs plain text mode...");

        // First in chat mode
        genParams.chatMode = true;
        genParams.prompt = "Hello!";
        genParams.nPredict = 50;
        genParams.dryMultiplier = 0.0f; // Disable DRY for this test

        System.out.println("Chat mode (with templates and history):");
        System.out.print("Response: ");

        result = LlamaCpp.llama_generate(handle, genParams);
        System.out.println();

        // Then in plain text mode
        genParams.chatMode = false;
        genParams.prompt = "Hello!";

        System.out.println("\nPlain text mode (no templates, no history):");
        System.out.print("Response: ");

        result = LlamaCpp.llama_generate(handle, genParams);
        System.out.println();

        // Test 7: Clear chat and test n_keep behavior
        System.out.println("\n\n7. Testing chat clear and n_keep behavior...");
        LlamaCpp.llama_clear_chat(handle);
        System.out.println("Chat history cleared");

        // Generate after clearing
        genParams.chatMode = false;
        genParams.prompt = "After clearing, the weather is";
        genParams.nPredict = 30;

        System.out.print("Response after clear: ");
        result = LlamaCpp.llama_generate(handle, genParams);
        System.out.println();

        // Test 8: Multiple init/destroy cycles
        System.out.println("\n\n8. Testing multiple init/destroy cycles...");
        for (int i = 0; i < 3; i++) {
            System.out.println("Cycle " + (i + 1) + "...");

            long testHandle = LlamaCpp.llama_init(initParams);
            if (testHandle == 0) {
                System.err.println("Failed to initialize in cycle " + (i + 1));
                break;
            }

            // Do a quick generation
            genParams.prompt = "Test " + i;
            genParams.nPredict = 10;
            genParams.tokenCallback = null; // No callback for this test

            result = LlamaCpp.llama_generate(testHandle, genParams);
            if (result != 0) {
                System.err.println("Generation failed in cycle " + (i + 1));
            }

            // Destroy
            LlamaCpp.llama_destroy(testHandle);
            System.out.println("Cycle " + (i + 1) + " completed");
        }

        // Clean up
        System.out.println("\n9. Cleaning up...");
        LlamaCpp.llama_destroy(handle);

        System.out.println("\n=== All tests completed ===");
    }
}
