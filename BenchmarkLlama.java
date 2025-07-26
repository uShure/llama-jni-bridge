import org.llm.wrapper.*;
import java.util.function.Predicate;

public class BenchmarkLlama {
    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: java BenchmarkLlama <model_path>");
            System.exit(1);
        }

        String modelPath = args[0];
        System.out.println("=== LLaMA JNI Bridge Benchmark ===");
        System.out.println("Model: " + modelPath);

        LlamaCpp llama = new LlamaCpp();

        // Test different context sizes
        int[] contextSizes = {512, 1024, 2048, 4096};

        for (int ctxSize : contextSizes) {
            System.out.println("\n--- Testing with context size: " + ctxSize + " ---");

            InitParams initParams = new InitParams();
            initParams.modelPath = modelPath;
            initParams.nCtx = ctxSize;
            initParams.nBatch = 512;
            initParams.nGpuLayers = 0;
            initParams.nThreads = Runtime.getRuntime().availableProcessors();

            // Measure model loading time
            long loadStart = System.currentTimeMillis();
            long handle = llama.llama_init(initParams);
            long loadTime = System.currentTimeMillis() - loadStart;

            if (handle == 0) {
                System.err.println("Failed to load model with context size " + ctxSize);
                continue;
            }

            System.out.println("Model loaded in: " + loadTime + "ms");

            // Test different prompt lengths
            String[] prompts = {
                "Hello",
                "The quick brown fox jumps over the lazy dog",
                "In a world where technology advances at an unprecedented pace, artificial intelligence has become"
            };

            for (String prompt : prompts) {
                System.out.println("\nPrompt length: " + prompt.length() + " chars");

                // Warm up
                runGeneration(llama, handle, prompt, 10);

                // Actual benchmark
                long totalTime = 0;
                int totalTokens = 0;
                int runs = 3;

                for (int i = 0; i < runs; i++) {
                    long genStart = System.currentTimeMillis();
                    int tokens = runGeneration(llama, handle, prompt, 50);
                    long genTime = System.currentTimeMillis() - genStart;

                    totalTime += genTime;
                    totalTokens += tokens;
                }

                double avgTime = totalTime / (double) runs;
                double avgTokens = totalTokens / (double) runs;
                double tokensPerSecond = (avgTokens * 1000) / avgTime;

                System.out.printf("Average: %.1f tokens in %.1fms (%.2f tokens/sec)\n",
                                avgTokens, avgTime, tokensPerSecond);
            }

            // Memory usage
            Runtime runtime = Runtime.getRuntime();
            long usedMemory = (runtime.totalMemory() - runtime.freeMemory()) / 1024 / 1024;
            System.out.println("\nMemory usage: " + usedMemory + " MB");

            llama.llama_destroy(handle);
        }

        System.out.println("\nBenchmark completed!");
    }

    private static int runGeneration(LlamaCpp llama, long handle, String prompt, int maxTokens) {
        GenerateParams genParams = new GenerateParams();
        genParams.prompt = prompt;
        genParams.nPredict = maxTokens;
        genParams.temp = 0.8f;
        genParams.topK = 40;
        genParams.topP = 0.95f;
        //genParams.stream = false;

        final int[] tokenCount = {0};
        genParams.tokenCallback = new Predicate<String>() {
            @Override
            public boolean test(String token) {
                tokenCount[0]++;
                return true;
            }
        };

        llama.llama_generate(handle, genParams);
        return tokenCount[0];
    }
}
