import org.llm.wrapper.*;
import java.util.function.Predicate;

public class Example {
    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: java Example <model_path> [prompt]");
            System.exit(1);
        }

        String modelPath = args[0];
        String prompt = args.length > 1 ? args[1] : "Once upon a time";

        LlamaCpp llama = new LlamaCpp();

        try {
            System.out.println("Loading model from: " + modelPath);

            // Initialize model
            InitParams initParams = new InitParams();
            initParams.modelPath = modelPath;
            initParams.nCtx = 2048;        // Context size
            initParams.nBatch = 512;       // Batch size
            initParams.nGpuLayers = 0;     // CPU only by default
            initParams.nThreads = 8;       // Number of CPU threads
            initParams.useMmap = true;     // Use memory mapping

            long handle = llama.llama_init(initParams);
            if (handle == 0) {
                System.err.println("Failed to initialize model");
                return;
            }

            System.out.println("Model loaded successfully!");
            System.out.println("Generating text for prompt: " + prompt);
            System.out.println("----------------------------------------");

            // Generate text
            GenerateParams genParams = new GenerateParams();
            genParams.prompt = prompt;
            genParams.nPredict = 100;      // Max tokens to generate
            genParams.temp = 0.8f;         // Temperature
            genParams.topK = 40;           // Top-K sampling
            genParams.topP = 0.95f;        // Top-P sampling
            genParams.repeatPenalty = 1.1f; // Repetition penalty
            genParams.stream = true;       // Enable streaming

            // Set up streaming callback
            genParams.tokenCallback = new Predicate<String>() {
                @Override
                public boolean test(String token) {
                    System.out.print(token);
                    System.out.flush();
                    return true; // Return false to stop generation
                }
            };

            int result = llama.llama_generate(handle, genParams);

            System.out.println("\n----------------------------------------");

            if (result != 0) {
                System.err.println("Generation failed with error code: " + result);
            } else {
                System.out.println("Generation completed successfully!");
            }

            // Clean up
            System.out.println("Cleaning up...");
            llama.llama_exit(handle);

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
