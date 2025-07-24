import org.llm.wrapper.*;
import java.util.Scanner;

public class ExampleUsage {
    // Load the native library
    static {
        System.loadLibrary("java_wrapper");
    }

    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: java ExampleUsage <model_path>");
            System.err.println("Example: java ExampleUsage models/llama-2-7b.Q4_K_M.gguf");
            System.exit(1);
        }

        String modelPath = args[0];
        Scanner scanner = new Scanner(System.in);

        System.out.println("=== LLaMA JNI Wrapper Example ===");
        System.out.println("Model: " + modelPath);
        System.out.println();

        // Initialize model
        System.out.println("Loading model...");
        InitParams initParams = new InitParams();
        initParams.modelPath = modelPath;
        initParams.nCtx = 2048;        // Context size
        initParams.nBatch = 512;       // Batch size
        initParams.nThreads = 4;       // CPU threads
        initParams.nGpuLayers = 0;     // GPU layers (0 = CPU only)
        initParams.seed = -1;          // Random seed

        long handle = LlamaCpp.llama_init(initParams);
        if (handle == 0) {
            System.err.println("Failed to initialize model!");
            System.exit(1);
        }

        System.out.println("Model loaded successfully!");
        System.out.println("Type 'quit' to exit");
        System.out.println();

        // Interactive loop
        while (true) {
            System.out.print("You: ");
            String prompt = scanner.nextLine();

            if (prompt.equalsIgnoreCase("quit")) {
                break;
            }

            System.out.print("Assistant: ");

            // Generate response
            GenerateParams genParams = new GenerateParams();
            genParams.prompt = prompt;
            genParams.nPredict = 512;       // Max tokens to generate
            genParams.temp = 0.8f;          // Temperature
            genParams.topK = 40;            // Top-K sampling
            genParams.topP = 0.95f;         // Top-P sampling
            genParams.repeatPenalty = 1.1f; // Repetition penalty
            genParams.repeatLastN = 64;     // Last N tokens for repeat penalty
            genParams.stream = true;        // Enable streaming
            genParams.seed = -1;            // Random seed

            // Token callback for streaming output
            genParams.tokenCallback = token -> {
                System.out.print(token);
                System.out.flush();
                return true; // Continue generation
            };

            int result = LlamaCpp.llama_generate(handle, genParams);
            if (result != 0) {
                System.err.println("\nError during generation!");
            }

            System.out.println("\n");
        }

        // Cleanup
        System.out.println("Cleaning up...");
        LlamaCpp.llama_exit(handle);
        scanner.close();

        System.out.println("Goodbye!");
    }
}
