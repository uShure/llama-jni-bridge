import org.llm.wrapper.*;
import java.util.function.Predicate;

public class SafeExample {
    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: java SafeExample <model_path> [prompt]");
            System.exit(1);
        }

        String modelPath = args[0];
        String prompt = args.length > 1 ? args[1] : "Once upon a time";

        LlamaCpp llama = new LlamaCpp();

        try {
            System.out.println("Loading model from: " + modelPath);

            // Initialize with safer parameters
            InitParams initParams = new InitParams();
            initParams.modelPath = modelPath;
            initParams.nCtx = 512;         // Smaller context
            initParams.nBatch = 128;       // Smaller batch
            initParams.nGpuLayers = 0;
            initParams.nThreads = 4;
            initParams.useMmap = true;
            initParams.useMlock = false;

            long handle = llama.llama_init(initParams);
            if (handle == 0) {
                System.err.println("Failed to initialize model");
                return;
            }

            System.out.println("Model loaded successfully!");
            System.out.println("Generating text for prompt: " + prompt);
            System.out.println("----------------------------------------");

            // Generate with limited tokens
            GenerateParams genParams = new GenerateParams();
            genParams.prompt = prompt;
            genParams.nPredict = 30;       // Only 30 tokens
            genParams.temp = 0.7f;
            genParams.topK = 40;
            genParams.topP = 0.95f;
            genParams.repeatPenalty = 1.1f;
            genParams.stream = true;

            // Count tokens for safety
            final int[] tokenCount = {0};
            final int maxTokens = 30;

            genParams.tokenCallback = new Predicate<String>() {
                @Override
                public boolean test(String token) {
                    System.out.print(token);
                    System.out.flush();
                    tokenCount[0]++;
                    // Stop if we hit the limit
                    return tokenCount[0] < maxTokens;
                }
            };

            int result = llama.llama_generate(handle, genParams);

            System.out.println("\n----------------------------------------");
            System.out.println("Generated " + tokenCount[0] + " tokens");

            if (result != 0) {
                System.err.println("Generation returned error code: " + result);
            }

            // Clean up
            System.out.println("Cleaning up...");
            llama.llama_destroy(handle);
            System.out.println("Done!");

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }

        // Force garbage collection to help with cleanup
        System.gc();

        // Small delay before exit
        try {
            Thread.sleep(100);
        } catch (InterruptedException e) {
            // Ignore
        }
    }
}
