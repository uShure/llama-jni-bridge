# 🛡️ Safety Features in llama-jni-bridge

## Overview
This JNI wrapper implements multiple safety mechanisms to ensure robust memory management and prevent crashes.

## 1. JNI Lifecycle Management

### JNI_OnLoad
```cpp
JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
    llama_backend_init();
    llama_numa_init(ggml_numa_strategy::GGML_NUMA_STRATEGY_DISABLED);
    return JNI_VERSION_1_8;
}
```
- Called when the library is loaded
- Initializes the llama backend once
- Sets up NUMA strategy

### JNI_OnUnload
```cpp
JNIEXPORT void JNICALL JNI_OnUnload(JavaVM *vm, void *reserved) {
    std::lock_guard<std::mutex> lock(g_context_registry_mutex);
    // Clean up any contexts the user forgot to destroy
    for (auto* data : g_active_contexts) {
        if (data->sampler) llama_sampler_free(data->sampler);
        if (data->ctx) llama_free(data->ctx);
        if (data->model) llama_model_free(data->model);
        delete data;
    }
    g_active_contexts.clear();
    llama_backend_free();
}
```
- Called when the library is unloaded (including unexpected JVM termination)
- Automatically cleans up all forgotten contexts
- Frees the backend resources

## 2. Context Registry & Validation

### Global Registry
```cpp
static std::mutex g_context_registry_mutex;
static std::unordered_set<LlamaContextData*> g_active_contexts;
```
- Thread-safe registry of all active contexts
- Protected by mutex for concurrent access

### Pointer Validation
```cpp
static bool is_valid_context(LlamaContextData* data) {
    std::lock_guard<std::mutex> lock(g_context_registry_mutex);
    return g_active_contexts.count(data) > 0;
}
```
- Every JNI method validates the context pointer before use
- Prevents use-after-free and invalid pointer access

### Usage Example
```cpp
JNIEXPORT jint JNICALL Java_org_llm_wrapper_LlamaCpp_llama_1generate
  (JNIEnv *env, jclass, jlong handle, jobject genParams) {
    LlamaContextData* data = reinterpret_cast<LlamaContextData*>(handle);
    if (!is_valid_context(data)) {
        last_error_message = "Invalid context handle";
        return -1;
    }
    // ... safe to use data here ...
}
```

## 3. Memory Management

### Context Creation
```cpp
// Register context
{
    std::lock_guard<std::mutex> lock(g_context_registry_mutex);
    g_active_contexts.insert(data_ptr);
}
```

### Context Destruction
```cpp
// Unregister context first (prevents use during destruction)
{
    std::lock_guard<std::mutex> lock(g_context_registry_mutex);
    g_active_contexts.erase(data);
}
// Then free resources
```

## 4. Native Memory Tracking

### Correct Memory Measurement
```cpp
JNIEXPORT jlong JNICALL Java_org_llm_wrapper_LlamaCpp_llama_1get_1native_1memory_1usage
  (JNIEnv *env, jclass, jlong handle) {
    // Returns actual native memory usage, not JVM heap
    size_t model_size = llama_model_size(data->model);
    size_t ctx_size = llama_state_get_size(data->ctx);
    return static_cast<jlong>(model_size + ctx_size);
}
```

## 5. Error Handling

### Thread-Local Error Messages
```cpp
thread_local std::string last_error_message;
```
- Each thread has its own error message
- Prevents race conditions in error reporting

### Error Propagation
```java
String error = LlamaCpp.llama_get_error();
if (error != null) {
    System.err.println("Error: " + error);
}
```

## 6. Best Practices for Users

### Always Use Try-Finally
```java
long handle = 0;
try {
    handle = LlamaCpp.llama_init(params);
    // Use the model
} finally {
    if (handle != 0) {
        LlamaCpp.llama_destroy(handle);
    }
}
```

### Shutdown Hooks
```java
Runtime.getRuntime().addShutdownHook(new Thread(() -> {
    if (modelHandle != 0) {
        LlamaCpp.llama_destroy(modelHandle);
    }
}));
```

## Summary
This implementation provides multiple layers of safety:
1. **Automatic cleanup** via JNI_OnUnload
2. **Pointer validation** before every use
3. **Thread-safe** context registry
4. **Proper memory tracking** from native side
5. **Graceful error handling**

Even if the JVM crashes or the user forgets to call destroy(), the resources will be properly cleaned up.
