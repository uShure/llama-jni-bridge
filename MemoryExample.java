import org.llm.wrapper.*;

public class MemoryExample {
    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: java MemoryExample <model_path>");
            System.exit(1);
        }

        String modelPath = args[0];
        LlamaCpp llama = new LlamaCpp();

        try {
            System.out.println("=== Memory Usage Example ===\n");

            // JVM memory before loading
            long jvmMemBefore = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
            System.out.printf("JVM memory before loading: %.2f MB\n", jvmMemBefore / 1024.0 / 1024.0);

            // Initialize model
            InitParams params = new InitParams();
            params.modelPath = modelPath;
            params.nCtx = 2048;
            params.nBatch = 512;
            params.nGpuLayers = 0;
            params.nThreads = 8;
            params.useMmap = true;
            params.seed = -1;

            System.out.println("\nLoading model...");
            long handle = llama.llama_init(params);

            if (handle == 0) {
                System.err.println("Failed to initialize model");
                String error = llama.llama_get_error();
                if (error != null) {
                    System.err.println("Error: " + error);
                }
                return;
            }

            // JVM memory after loading
            long jvmMemAfter = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
            System.out.printf("\nJVM memory after loading: %.2f MB\n", jvmMemAfter / 1024.0 / 1024.0);
            System.out.printf("JVM memory increase: %.2f MB (INCORRECT - doesn't include native memory)\n",
                (jvmMemAfter - jvmMemBefore) / 1024.0 / 1024.0);

            // Native memory usage (CORRECT measurement)
            long nativeMemory = llama.llama_get_native_memory_usage(handle);
            if (nativeMemory > 0) {
                System.out.printf("\nNative memory usage: %.2f MB (CORRECT - actual model size)\n",
                    nativeMemory / 1024.0 / 1024.0);
            } else {
                System.out.println("\nFailed to get native memory usage");
            }

            System.out.println("\nAs you can see, JVM memory tracking doesn't show the actual");
            System.out.println("memory used by the native library. Use llama_get_native_memory_usage()");
            System.out.println("to get the correct memory consumption.");

            // Cleanup
            llama.llama_destroy(handle);
            System.out.println("\nModel destroyed successfully");

        } catch (UnsatisfiedLinkError e) {
            System.err.println("Native library not found!");
            System.err.println("Make sure the JNI library is built and in your library path.");
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
