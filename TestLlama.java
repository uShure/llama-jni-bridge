import org.llm.wrapper.*;
import java.util.function.Predicate;
import java.util.Scanner;

public class TestLlama {
    private static final String ANSI_RESET = "\u001B[0m";
    private static final String ANSI_GREEN = "\u001B[32m";
    private static final String ANSI_YELLOW = "\u001B[33m";
    private static final String ANSI_BLUE = "\u001B[34m";
    private static final String ANSI_RED = "\u001B[31m";

    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: java TestLlama <model_path>");
            System.exit(1);
        }

        String modelPath = args[0];
        System.out.println(ANSI_BLUE + "=== LLaMA JNI Bridge Test ===" + ANSI_RESET);
        System.out.println("Model: " + modelPath);

        LlamaCpp llama = new LlamaCpp();
        long handle = 0;

        try {
            // Test 1: Model initialization
            System.out.println("\n" + ANSI_YELLOW + "Test 1: Model Initialization" + ANSI_RESET);
            InitParams initParams = new InitParams();
            initParams.modelPath = modelPath;
            initParams.nCtx = 2048;
            initParams.nBatch = 512;
            initParams.nGpuLayers = 0; // CPU only for compatibility
            initParams.nThreads = 8;
            initParams.useMmap = true;
            initParams.useMlock = false;

            long startTime = System.currentTimeMillis();
            handle = llama.llama_init(initParams);
            long loadTime = System.currentTimeMillis() - startTime;

            if (handle == 0) {
                System.err.println(ANSI_RED + "FAILED: Could not initialize model" + ANSI_RESET);
                System.exit(1);
            }

            System.out.println(ANSI_GREEN + "SUCCESS: Model loaded in " + loadTime + "ms" + ANSI_RESET);

            // Test 2: Simple generation
            System.out.println("\n" + ANSI_YELLOW + "Test 2: Simple Text Generation" + ANSI_RESET);
            testGeneration(llama, handle, "Once upon a time", 50);

            // Test 3: Different parameters
            System.out.println("\n" + ANSI_YELLOW + "Test 3: Generation with Different Parameters" + ANSI_RESET);
            testGenerationWithParams(llama, handle);

            // Test 4: Interactive mode
            System.out.println("\n" + ANSI_YELLOW + "Test 4: Interactive Mode" + ANSI_RESET);
            System.out.println("Enter prompts (type 'quit' to exit):");

            Scanner scanner = new Scanner(System.in);
            while (true) {
                System.out.print("\n> ");
                String prompt = scanner.nextLine();
                if (prompt.equalsIgnoreCase("quit")) break;

                GenerateParams genParams = new GenerateParams();
                genParams.prompt = prompt;
                genParams.nPredict = 100;
                genParams.temp = 0.8f;
                genParams.topK = 40;
                genParams.topP = 0.95f;
                genParams.repeatPenalty = 1.1f;
                genParams.stream = true;

                final StringBuilder response = new StringBuilder();
                genParams.tokenCallback = new Predicate<String>() {
                    @Override
                    public boolean test(String token) {
                        System.out.print(token);
                        System.out.flush();
                        response.append(token);
                        return true;
                    }
                };

                System.out.println(ANSI_BLUE);
                llama.llama_generate(handle, genParams);
                System.out.println(ANSI_RESET);
            }

            System.out.println(ANSI_GREEN + "\nAll tests completed successfully!" + ANSI_RESET);

        } catch (Exception e) {
            System.err.println(ANSI_RED + "Error: " + e.getMessage() + ANSI_RESET);
            e.printStackTrace();
        } finally {
            if (handle != 0) {
                System.out.println("\nCleaning up...");
                llama.llama_destroy(handle);
            }
        }
    }

    private static void testGeneration(LlamaCpp llama, long handle, String prompt, int maxTokens) {
        System.out.println("Prompt: \"" + prompt + "\"");
        System.out.println("Generating " + maxTokens + " tokens...\n");

        GenerateParams genParams = new GenerateParams();
        genParams.prompt = prompt;
        genParams.nPredict = maxTokens;
        genParams.temp = 0.8f;
        genParams.topK = 40;
        genParams.topP = 0.95f;
        genParams.repeatPenalty = 1.1f;
        genParams.stream = true;

        final int[] tokenCount = {0};
        genParams.tokenCallback = new Predicate<String>() {
            @Override
            public boolean test(String token) {
                System.out.print(token);
                System.out.flush();
                tokenCount[0]++;
                return true;
            }
        };

        long startTime = System.currentTimeMillis();
        int result = llama.llama_generate(handle, genParams);
        long genTime = System.currentTimeMillis() - startTime;

        System.out.println("\n");
        if (result == 0) {
            double tokensPerSecond = (tokenCount[0] * 1000.0) / genTime;
            System.out.println(ANSI_GREEN + "SUCCESS: Generated " + tokenCount[0] +
                             " tokens in " + genTime + "ms (" +
                             String.format("%.2f", tokensPerSecond) + " tokens/sec)" + ANSI_RESET);
        } else {
            System.err.println(ANSI_RED + "FAILED: Generation error code " + result + ANSI_RESET);
        }
    }

    private static void testGenerationWithParams(LlamaCpp llama, long handle) {
        // Test with low temperature (more deterministic)
        System.out.println("\nLow temperature (0.3) - More deterministic:");
        GenerateParams genParams1 = new GenerateParams();
        genParams1.prompt = "The capital of France is";
        genParams1.nPredict = 20;
        genParams1.temp = 0.3f;
        genParams1.topK = 10;
        genParams1.stream = true;
        genParams1.tokenCallback = token -> {
            System.out.print(token);
            return true;
        };
        llama.llama_generate(handle, genParams1);
        System.out.println();

        // Test with high temperature (more creative)
        System.out.println("\nHigh temperature (1.5) - More creative:");
        GenerateParams genParams2 = new GenerateParams();
        genParams2.prompt = "The capital of France is";
        genParams2.nPredict = 20;
        genParams2.temp = 1.5f;
        genParams2.topK = 100;
        genParams2.stream = true;
        genParams2.tokenCallback = token -> {
            System.out.print(token);
            return true;
        };
        llama.llama_generate(handle, genParams2);
        System.out.println();
    }
}
