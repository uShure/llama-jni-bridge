import org.llm.wrapper.*;

public class SafeShutdownExample {
    static {
        System.loadLibrary("java_wrapper");
    }

    private static volatile long modelHandle = 0;

    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: java SafeShutdownExample <model_path>");
            System.exit(1);
        }

        // Add shutdown hook for cleanup
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("\nShutting down...");
            if (modelHandle != 0) {
                try {
                    LlamaCpp.llama_destroy(modelHandle);
                    modelHandle = 0;
                } catch (Exception e) {
                    System.err.println("Error during cleanup: " + e.getMessage());
                }
            }
        }));

        try {
            // Initialize model
            System.out.println("Loading model...");
            InitParams init = new InitParams();
            init.modelPath = args[0];
            init.nCtx = 2048;
            init.nBatch = 512;
            init.nThreads = 4;
            init.nGpuLayers = 0; // CPU only for testing

            modelHandle = LlamaCpp.llama_init(init);
            if (modelHandle == 0) {
                System.err.println("Failed to initialize model!");
                System.exit(1);
            }

            System.out.println("Model loaded successfully!");

            // Generate text
            GenerateParams gen = new GenerateParams();
            gen.prompt = "The capital of France is";
            gen.nPredict = 50;
            gen.temp = 0.8f;
            gen.topK = 40;
            gen.topP = 0.95f;
            gen.repeatPenalty = 1.1f;
            gen.repeatLastN = 64;
            //gen.stream = true;
            gen.tokenCallback = token -> {
                System.out.print(token);
                System.out.flush();
                return true;
            };

            System.out.print("Response: " + gen.prompt);
            int result = LlamaCpp.llama_generate(modelHandle, gen);
            System.out.println();

            if (result != 0) {
                System.err.println("Generation failed!");
            }

            // Clean up explicitly
            System.out.println("\nCleaning up...");
            LlamaCpp.llama_destroy(modelHandle);
            modelHandle = 0;

            System.out.println("Done!");

            // Force exit to avoid hanging
            // Remove this line after testing if the hang is fixed
            System.exit(0);

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}
