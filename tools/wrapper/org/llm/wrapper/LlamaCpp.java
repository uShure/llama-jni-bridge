package org.llm.wrapper;

public class LlamaCpp {
    static {
        // On Mac, this will be libjava_wrapper.dylib
        // On Linux, libjava_wrapper.so
        System.loadLibrary("java_wrapper");
    }

    /**
     * Initializes the llama.cpp backend and loads a model.
     * @param initParams Parameters for initialization.
     * @return A handle (pointer) to the created context, or 0 on failure.
     */
    public static native long llama_init(InitParams params);

    /**
     * Frees all resources associated with a model context.
     * @param handle The handle returned by llama_init.
     */
    public static native void llama_destroy(long handle);

    /**
     * Generates text from a prompt.
     * @param handle The handle returned by llama_init.
     * @param generateParams Parameters for generation.
     * @return 0 on success, non-zero on failure.
     */
    public static native int llama_generate(long handle, GenerateParams params);

    public static native String llama_get_error();
    public static native long llama_get_native_memory_usage(long handle);

    /**
     * Gets the OS-reported memory usage for the process.
     * @return Memory usage in bytes, or -1 on error.
     */
    public static native long llama_get_os_memory_usage();

    /**
     * Clears the chat history and KV cache for a context.
     * @param handle The handle returned by llama_init.
     */
    public static native void llama_clear_chat(long handle);
}
