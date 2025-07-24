import org.llm.wrapper.*;

public class BatchExample {
    static {
        System.loadLibrary("java_wrapper");
    }

    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: java BatchExample <model_path>");
            System.exit(1);
        }

        // Initialize model
        InitParams init = new InitParams();
        init.modelPath = args[0];
        init.nCtx = 2048;
        init.nBatch = 512;
        init.nThreads = 4;

        long handle = LlamaCpp.llama_init(init);
        if (handle == 0) {
            System.err.println("Failed to load model!");
            System.exit(1);
        }

        // Test prompts
        String[] prompts = {
            "The capital of France is",
            "The largest planet in our solar system is",
            "Water freezes at",
            "The speed of light is approximately",
            "The author of Romeo and Juliet is"
        };

        // Generate responses
        for (String prompt : prompts) {
            System.out.println("Prompt: " + prompt);
            System.out.print("Response: " + prompt);

            GenerateParams gen = new GenerateParams();
            gen.prompt = prompt;
            gen.nPredict = 50;
            gen.temp = 0.7f;
            gen.topK = 40;
            gen.topP = 0.9f;
            gen.stream = true;
            gen.tokenCallback = token -> {
                System.out.print(token);
                System.out.flush();
                return true;
            };

            LlamaCpp.llama_generate(handle, gen);
            System.out.println("\n");
        }

        // Cleanup
        LlamaCpp.llama_exit(handle);
    }
}
