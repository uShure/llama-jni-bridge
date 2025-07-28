#include "org_llm_wrapper_LlamaCpp.h"
#include "llama.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include <iostream>
#include <string>
#include <vector>
#include <unordered_set>
#include <mutex>
#include <memory>
#include <thread>
#include <cstring>
#include <sstream>
#include <fstream>
#include <nlohmann/json.hpp>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#elif __APPLE__
#include <mach/mach.h>
#include <mach/task.h>
#else // Linux
#include <unistd.h>
#include <fstream>
#endif

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#endif

// --- Context Data Structure ---

struct LlamaContextData {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    llama_sampler * sampler = nullptr;
    common_params params;
    int n_ctx = 0;
    std::vector<llama_chat_message> chat_messages;  // Store chat history
    std::vector<char> formatted_chat;  // Store formatted chat

    // Token tracking for proper cache reuse
    std::vector<llama_token> all_tokens;  // All tokens ever processed
    int n_past = 0;  // Number of tokens in KV cache
    int n_keep = 0;  // Number of tokens to keep from initial prompt

    // For proper caching in plain text mode
    std::string last_prompt;  // Track previous prompt to detect continuations
};

// --- Globals for safe resource management ---
static std::mutex g_context_registry_mutex;
static std::unordered_set<LlamaContextData*> g_active_contexts;

// --- Error handling ---
static std::mutex g_error_mutex;
static std::string last_error_message;

// --- Helper function to free chat messages ---
static void free_chat_messages(std::vector<llama_chat_message>& messages) {
    for (auto & msg : messages) {
        if (msg.content) {
            free(const_cast<char *>(msg.content));
            msg.content = nullptr;
        }
    }
    messages.clear();
}

// --- Helper function for error handling ---
static void set_last_error(const std::string& error) {
    std::lock_guard<std::mutex> lock(g_error_mutex);
    last_error_message = error;
}

// --- JNI Lifecycle Hooks ---

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM * /* vm */, void * /* reserved */) {
    // only print errors (like simple-chat.cpp)
    llama_log_set([](enum ggml_log_level level, const char * text, void * /* user_data */) {
        if (level >= GGML_LOG_LEVEL_ERROR) {
            fprintf(stderr, "%s", text);
        }
    }, nullptr);

    // load dynamic backends
    ggml_backend_load_all();

    llama_backend_init();
    llama_numa_init(ggml_numa_strategy::GGML_NUMA_STRATEGY_DISABLED);
    last_error_message = "No error";
    return JNI_VERSION_1_8;
}

JNIEXPORT void JNICALL JNI_OnUnload(JavaVM * /* vm */, void * /* reserved */) {
    std::lock_guard<std::mutex> lock(g_context_registry_mutex);
    // Clean up any contexts the user forgot to destroy
    for (auto* data : g_active_contexts) {
        // Free chat messages using helper function
        free_chat_messages(data->chat_messages);
        if (data->sampler) llama_sampler_free(data->sampler);
        if (data->ctx) llama_free(data->ctx);
        if (data->model) llama_model_free(data->model);
        delete data;
    }
    g_active_contexts.clear();
    llama_backend_free();
}

#ifdef __cplusplus
}
#endif

// --- Helper Functions ---

static bool is_valid_context(LlamaContextData* data) {
    std::lock_guard<std::mutex> lock(g_context_registry_mutex);
    return g_active_contexts.count(data) > 0;
}

static jint getIntField(JNIEnv *env, jobject obj, const char* fieldName) {
    jclass cls = env->GetObjectClass(obj);
    if (!cls) {
        fprintf(stderr, "ERROR: Failed to get class for field %s\n", fieldName);
        return 0;
    }

    jfieldID fid = env->GetFieldID(cls, fieldName, "I");
    if (!fid) {
        // Clear exception if any
        if (env->ExceptionCheck()) {
            env->ExceptionDescribe();
            env->ExceptionClear();
        }
        fprintf(stderr, "WARNING: Field %s (I) not found, using default value 0\n", fieldName);
        return 0;
    }

    return env->GetIntField(obj, fid);
}

static jfloat getFloatField(JNIEnv *env, jobject obj, const char* fieldName) {
    jclass cls = env->GetObjectClass(obj);
    if (!cls) {
        fprintf(stderr, "ERROR: Failed to get class for field %s\n", fieldName);
        return 0.0f;
    }

    jfieldID fid = env->GetFieldID(cls, fieldName, "F");
    if (!fid) {
        // Clear exception if any
        if (env->ExceptionCheck()) {
            env->ExceptionDescribe();
            env->ExceptionClear();
        }
        fprintf(stderr, "WARNING: Field %s (F) not found, using default value 0.0\n", fieldName);
        return 0.0f;
    }

    return env->GetFloatField(obj, fid);
}

static jboolean getBooleanField(JNIEnv *env, jobject obj, const char* fieldName) {
    jclass cls = env->GetObjectClass(obj);
    if (!cls) {
        fprintf(stderr, "ERROR: Failed to get class for field %s\n", fieldName);
        return JNI_FALSE;
    }

    jfieldID fid = env->GetFieldID(cls, fieldName, "Z");
    if (!fid) {
        // Clear exception if any
        if (env->ExceptionCheck()) {
            env->ExceptionDescribe();
            env->ExceptionClear();
        }
        fprintf(stderr, "WARNING: Field %s (Z) not found, using default value false\n", fieldName);
        return JNI_FALSE;
    }

    return env->GetBooleanField(obj, fid);
}

static jstring getStringField(JNIEnv *env, jobject obj, const char* fieldName) {
    jclass cls = env->GetObjectClass(obj);
    if (!cls) {
        fprintf(stderr, "ERROR: Failed to get class for field %s\n", fieldName);
        return nullptr;
    }

    jfieldID fid = env->GetFieldID(cls, fieldName, "Ljava/lang/String;");
    if (!fid) {
        // Clear exception if any
        if (env->ExceptionCheck()) {
            env->ExceptionDescribe();
            env->ExceptionClear();
        }
        fprintf(stderr, "WARNING: Field %s (Ljava/lang/String;) not found, using null\n", fieldName);
        return nullptr;
    }

    return (jstring)env->GetObjectField(obj, fid);
}

static jobject getObjectField(JNIEnv *env, jobject obj, const char* fieldName, const char* sig) {
    jclass cls = env->GetObjectClass(obj);
    jfieldID fid = env->GetFieldID(cls, fieldName, sig);
    return env->GetObjectField(obj, fid);
}

static jintArray getIntArrayField(JNIEnv *env, jobject obj, const char* fieldName) {
    jclass cls = env->GetObjectClass(obj);
    if (!cls) {
        fprintf(stderr, "ERROR: Failed to get class for field %s\n", fieldName);
        return nullptr;
    }

    jfieldID fid = env->GetFieldID(cls, fieldName, "[I");
    if (!fid) {
        // Clear exception if any
        if (env->ExceptionCheck()) {
            env->ExceptionDescribe();
            env->ExceptionClear();
        }
        fprintf(stderr, "WARNING: Field %s ([I) not found, using null\n", fieldName);
        return nullptr;
    }

    return (jintArray)env->GetObjectField(obj, fid);
}

static jfloatArray getFloatArrayField(JNIEnv *env, jobject obj, const char* fieldName) {
    jclass cls = env->GetObjectClass(obj);
    if (!cls) {
        fprintf(stderr, "ERROR: Failed to get class for field %s\n", fieldName);
        return nullptr;
    }

    jfieldID fid = env->GetFieldID(cls, fieldName, "[F");
    if (!fid) {
        // Clear exception if any
        if (env->ExceptionCheck()) {
            env->ExceptionDescribe();
            env->ExceptionClear();
        }
        fprintf(stderr, "WARNING: Field %s ([F) not found, using null\n", fieldName);
        return nullptr;
    }

    return (jfloatArray)env->GetObjectField(obj, fid);
}

// --- OS Memory Measurement ---
static size_t get_process_memory_usage() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize;
    }
    return 0;
#elif __APPLE__
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &infoCount) == KERN_SUCCESS) {
        return info.resident_size;
    }
    return 0;
#else // Linux
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.find("VmRSS:") == 0) {
            size_t pos = line.find_first_of("0123456789");
            if (pos != std::string::npos) {
                return std::stoull(line.substr(pos)) * 1024; // Convert KB to bytes
            }
        }
    }
    return 0;
#endif
}

// --- JNI Implementations ---

JNIEXPORT jlong JNICALL Java_org_llm_wrapper_LlamaCpp_llama_1init
  (JNIEnv *env, jclass, jobject initParams) {
    auto data = std::make_unique<LlamaContextData>();

    const char* modelPath_cstr = nullptr;
    jstring modelPath_jstr = getStringField(env, initParams, "modelPath");
    if (modelPath_jstr) {
        modelPath_cstr = env->GetStringUTFChars(modelPath_jstr, nullptr);
    }

    if (!modelPath_cstr) {
        return 0; // Model path is mandatory
    }

    // Extract ALL parameters from InitParams

    // === Threading and CPU Configuration ===
    int n_threads = getIntField(env, initParams, "nThreads");
    data->params.cpuparams.n_threads = (n_threads == -1) ? std::thread::hardware_concurrency() : n_threads;

    int n_threads_batch = getIntField(env, initParams, "nThreadsBatch");
    data->params.cpuparams_batch.n_threads = (n_threads_batch == -1) ? data->params.cpuparams.n_threads : n_threads_batch;

    // TODO: CPU affinity parameters (cpuMask, cpuRange, etc.) - requires platform-specific code

    // === CPU Affinity Configuration ===
    jstring cpu_mask_str = getStringField(env, initParams, "cpuMask");
    jstring cpu_range_str = getStringField(env, initParams, "cpuRange");
    bool cpu_strict = getBooleanField(env, initParams, "cpuStrict");
    int priority = getIntField(env, initParams, "priority");
    int poll = getIntField(env, initParams, "poll");

    // Batch CPU affinity
    jstring cpu_mask_batch_str = getStringField(env, initParams, "cpuMaskBatch");
    jstring cpu_range_batch_str = getStringField(env, initParams, "cpuRangeBatch");
    bool cpu_strict_batch = getBooleanField(env, initParams, "cpuStrictBatch");
    int priority_batch = getIntField(env, initParams, "priorityBatch");
    int poll_batch = getIntField(env, initParams, "pollBatch");

    // TODO: Apply CPU affinity settings when platform-specific code is available
    if (cpu_mask_str) {
        const char* mask_cstr = env->GetStringUTFChars(cpu_mask_str, nullptr);
        // TODO: Parse and apply CPU mask
        env->ReleaseStringUTFChars(cpu_mask_str, mask_cstr);
    }
    if (cpu_range_str) {
        const char* range_cstr = env->GetStringUTFChars(cpu_range_str, nullptr);
        // TODO: Parse and apply CPU range
        env->ReleaseStringUTFChars(cpu_range_str, range_cstr);
    }
    (void)cpu_strict;
    (void)priority;
    (void)poll;

    if (cpu_mask_batch_str) {
        const char* mask_cstr = env->GetStringUTFChars(cpu_mask_batch_str, nullptr);
        // TODO: Parse and apply batch CPU mask
        env->ReleaseStringUTFChars(cpu_mask_batch_str, mask_cstr);
    }
    if (cpu_range_batch_str) {
        const char* range_cstr = env->GetStringUTFChars(cpu_range_batch_str, nullptr);
        // TODO: Parse and apply batch CPU range
        env->ReleaseStringUTFChars(cpu_range_batch_str, range_cstr);
    }
    (void)cpu_strict_batch;
    (void)priority_batch;
    (void)poll_batch;

    // === Context and Batch Configuration ===
    data->params.n_ctx = getIntField(env, initParams, "nCtx");
    data->params.n_batch = getIntField(env, initParams, "nBatch");
    data->params.n_ubatch = getIntField(env, initParams, "nUBatch");

    // Match simple-chat.cpp: if n_batch is 0, set it to n_ctx
    if (data->params.n_batch == 0) {
        data->params.n_batch = data->params.n_ctx;
    }

    // Cache reuse parameter
    int n_keep = getIntField(env, initParams, "nKeep");
    data->n_keep = n_keep;  // Store initial n_keep value
    bool swa_full = getBooleanField(env, initParams, "swaFull");

    // === Model Loading Parameters ===
    bool use_mmap = getBooleanField(env, initParams, "useMmap");
    bool use_mlock = getBooleanField(env, initParams, "useMlock");
    bool check_tensors = getBooleanField(env, initParams, "checkTensors");
    bool embeddings_mode = getBooleanField(env, initParams, "embeddings");

    // === GPU Configuration ===
    int n_gpu_layers = getIntField(env, initParams, "nGpuLayers");
    int split_mode = getIntField(env, initParams, "splitMode");
    int main_gpu = getIntField(env, initParams, "mainGpu");

    // Array to store tensor splits - declared at this scope so it remains valid
    float tensor_split_arr[128] = {0}; // Initialize all to 0

    // Parse tensor split string
    jstring tensor_split_str = getStringField(env, initParams, "tensorSplit");
    std::vector<float> tensor_split_vec;
    if (tensor_split_str) {
        const char* splits_cstr = env->GetStringUTFChars(tensor_split_str, nullptr);
        if (splits_cstr && strlen(splits_cstr) > 0) {
            std::stringstream ss(splits_cstr);
            std::string token;
            while (std::getline(ss, token, ',')) {
                try {
                    tensor_split_vec.push_back(std::stof(token));
                } catch (...) {
                    // Ignore invalid values
                }
            }
        }
        env->ReleaseStringUTFChars(tensor_split_str, splits_cstr);
    }

    // === Performance Features ===
    bool flash_attn = getBooleanField(env, initParams, "flashAttn");
    bool no_kv_offload = getBooleanField(env, initParams, "noKVOffload");
    bool no_op_offload = getBooleanField(env, initParams, "noOpOffload");
    bool no_perf = getBooleanField(env, initParams, "noPerf");

    // === RoPE Configuration ===
    int rope_scaling_type = getIntField(env, initParams, "ropeScalingType");
    float rope_scale = getFloatField(env, initParams, "ropeScale");
    float rope_freq_base = getFloatField(env, initParams, "ropeFreqBase");
    float rope_freq_scale = getFloatField(env, initParams, "ropeFreqScale");

    // Apply rope_scale by adjusting rope_freq_scale
    if (rope_scale > 0.0f) {
        // rope_scale is inverse of freq_scale
        // If user provides both rope_scale and rope_freq_scale, rope_scale takes precedence
        rope_freq_scale = 1.0f / rope_scale;
        // Note: verbose is not available here, it's in GenerateParams
        fprintf(stderr, "Applied rope_scale %.2f as rope_freq_scale %.2f\n", rope_scale, rope_freq_scale);
    }

    // === YaRN Parameters ===
    int yarn_orig_ctx = getIntField(env, initParams, "yarnOrigCtx");
    float yarn_ext_factor = getFloatField(env, initParams, "yarnExtFactor");
    float yarn_attn_factor = getFloatField(env, initParams, "yarnAttnFactor");
    float yarn_beta_slow = getFloatField(env, initParams, "yarnBetaSlow");
    float yarn_beta_fast = getFloatField(env, initParams, "yarnBetaFast");

    // === Cache Configuration ===
    int cache_type_k = getIntField(env, initParams, "cacheTypeK");
    int cache_type_v = getIntField(env, initParams, "cacheTypeV");
    int cache_type_k_draft = getIntField(env, initParams, "cacheTypeKDraft");
    int cache_type_v_draft = getIntField(env, initParams, "cacheTypeVDraft");
    float defrag_threshold = getFloatField(env, initParams, "defragThreshold");

    // TODO: Apply cache_type_k_draft and cache_type_v_draft when draft model support is added
    (void)cache_type_k_draft;
    (void)cache_type_v_draft;

    // === NUMA Configuration ===
    int numa_mode = getIntField(env, initParams, "numa");

    // === Escape processing ===
    bool no_escape = getBooleanField(env, initParams, "noEscape");
    data->params.escape = !no_escape;

    // === Sampling seed ===
    int seed = getIntField(env, initParams, "seed");
    // Fix: Only use default seed when user specifies -1
    if (seed == -1) {
        data->params.sampling.seed = LLAMA_DEFAULT_SEED;
    } else {
        data->params.sampling.seed = seed;
    }

    data->n_ctx = data->params.n_ctx;

    // === Additional InitParams fields ===
    // Sequences configuration
    int n_seq_max = getIntField(env, initParams, "nSeqMax");
    int n_parallel = getIntField(env, initParams, "nParallel");
    int n_sequences = getIntField(env, initParams, "nSequences");

    // TODO: Apply these parameters when API supports them
    (void)n_seq_max;
    (void)n_parallel;
    (void)n_sequences;

    // Draft model for speculative decoding
    jstring draft_model_str = getStringField(env, initParams, "draftModel");
    if (draft_model_str) {
        const char* draft_cstr = env->GetStringUTFChars(draft_model_str, nullptr);
        // TODO: Load draft model for speculative decoding
        env->ReleaseStringUTFChars(draft_model_str, draft_cstr);
    }

    // Logging configuration
    bool log_disable = getBooleanField(env, initParams, "logDisable");
    if (log_disable) {
        // Disable logging
        llama_log_set(nullptr, nullptr);
    }

    // GPU device configuration
    jstring device_str = getStringField(env, initParams, "device");
    if (device_str) {
        const char* device_cstr = env->GetStringUTFChars(device_str, nullptr);
        // Fix: device field no longer exists in common_params
        // This is now handled through model and context params
        // TODO: Handle device configuration through proper API
        env->ReleaseStringUTFChars(device_str, device_cstr);
    }

    // === LORA Configuration ===
    jstring lora_base_str = getStringField(env, initParams, "loraBase");
    if (lora_base_str) {
        const char* lora_base_cstr = env->GetStringUTFChars(lora_base_str, nullptr);
        // TODO: Set LORA base model when API supports it
        env->ReleaseStringUTFChars(lora_base_str, lora_base_cstr);
    }

    // Parse LORA adapters
    jstring lora_str = getStringField(env, initParams, "lora");
    jstring lora_scaled_str = getStringField(env, initParams, "loraScaled");
    if (lora_str) {
        const char* lora_cstr = env->GetStringUTFChars(lora_str, nullptr);
        // Format: "adapter1.gguf,adapter2.gguf"
        // TODO: Load LORA adapters after context creation
        env->ReleaseStringUTFChars(lora_str, lora_cstr);
    }
    if (lora_scaled_str) {
        const char* lora_scaled_cstr = env->GetStringUTFChars(lora_scaled_str, nullptr);
        // Format: "adapter1.gguf:0.5,adapter2.gguf:1.0"
        // TODO: Load LORA adapters with scales after context creation
        env->ReleaseStringUTFChars(lora_scaled_str, lora_scaled_cstr);
    }
    float lora_scale = getFloatField(env, initParams, "loraScale");
    (void)lora_scale; // Default scale for unscaled adapters

    // === Control Vector Configuration ===
    jstring control_vector_str = getStringField(env, initParams, "controlVector");
    jstring control_vector_scaled_str = getStringField(env, initParams, "controlVectorScaled");
    int control_vector_layer_start = getIntField(env, initParams, "controlVectorLayerStart");
    int control_vector_layer_end = getIntField(env, initParams, "controlVectorLayerEnd");

    if (control_vector_str) {
        const char* cv_cstr = env->GetStringUTFChars(control_vector_str, nullptr);
        // TODO: Load control vector after context creation
        env->ReleaseStringUTFChars(control_vector_str, cv_cstr);
    }
    if (control_vector_scaled_str) {
        const char* cv_scaled_cstr = env->GetStringUTFChars(control_vector_scaled_str, nullptr);
        // Format: "vector1.gguf:0.5,vector2.gguf:1.0"
        // TODO: Load control vectors with scales
        env->ReleaseStringUTFChars(control_vector_scaled_str, cv_scaled_cstr);
    }
    (void)control_vector_layer_start;
    (void)control_vector_layer_end;

    // === Extended Logging Configuration ===
    jstring log_file_str = getStringField(env, initParams, "logFile");
    bool log_prefix = getBooleanField(env, initParams, "logPrefix");
    bool log_timestamps = getBooleanField(env, initParams, "logTimestamps");

    if (log_file_str && !log_disable) {
        const char* log_file_cstr = env->GetStringUTFChars(log_file_str, nullptr);
        // TODO: Set up file logging
        // Would need to implement custom log handler that writes to file
        env->ReleaseStringUTFChars(log_file_str, log_file_cstr);
    }
    (void)log_prefix;
    (void)log_timestamps;

    // === Model Modification Parameters ===
    int quantize_output_tensor = getIntField(env, initParams, "quantizeOutputTensor");
    int only_write_model_tensor = getIntField(env, initParams, "onlyWriteModelTensor");

    // Override key-value pairs for model metadata
    jstring override_kv_str = getStringField(env, initParams, "overrideKv");
    if (override_kv_str) {
        const char* kv_cstr = env->GetStringUTFChars(override_kv_str, nullptr);
        // Format: "key1=value1,key2=value2"
        // TODO: Parse and apply KV overrides
        env->ReleaseStringUTFChars(override_kv_str, kv_cstr);
    }

    // Override tensor values
    jstring override_tensor_str = getStringField(env, initParams, "overrideTensor");
    if (override_tensor_str) {
        const char* tensor_cstr = env->GetStringUTFChars(override_tensor_str, nullptr);
        // TODO: Parse and apply tensor overrides
        env->ReleaseStringUTFChars(override_tensor_str, tensor_cstr);
    }

    (void)quantize_output_tensor;
    (void)only_write_model_tensor;

    // === Other Configuration ===
    bool use_nn_api = getBooleanField(env, initParams, "useNnApi");
    int n_gqa = getIntField(env, initParams, "nGqa");
    int n_chunks = getIntField(env, initParams, "nChunks");

    // TODO: Apply these parameters when API supports them
    (void)use_nn_api;
    (void)n_gqa;
    (void)n_chunks;

    // Load model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    model_params.use_mmap = use_mmap;
    model_params.use_mlock = use_mlock;
    model_params.check_tensors = check_tensors;
    model_params.main_gpu = main_gpu;
    model_params.split_mode = (llama_split_mode)split_mode;

    // Apply tensor splits if provided
    if (!tensor_split_vec.empty()) {
        // Apply tensor splits to model params
        // llama.cpp expects normalized values that sum to 1.0
        float sum = 0.0f;
        for (float val : tensor_split_vec) {
            sum += val;
        }

        // Normalize if needed
        if (sum > 0.0f && sum != 1.0f) {
            for (float& val : tensor_split_vec) {
                val /= sum;
            }
        }

        // Copy to our tensor_split array
        size_t n_splits = std::min(tensor_split_vec.size(), (size_t)128); // Typical max devices
        for (size_t i = 0; i < n_splits; i++) {
            tensor_split_arr[i] = tensor_split_vec[i];
        }

        // Point model_params.tensor_split to our array
        model_params.tensor_split = tensor_split_arr;

        // verbose is not available here, but we can still print info
        fprintf(stderr, "Applied tensor splits for %zu devices\n", n_splits);
    }

    data->model = llama_model_load_from_file(modelPath_cstr, model_params);
    env->ReleaseStringUTFChars(modelPath_jstr, modelPath_cstr);

    if (!data->model) {
        std::cerr << "Failed to load model" << std::endl;
        set_last_error("Failed to load model from file");
        return 0;
    }

    // Create context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = data->params.n_ctx;
    ctx_params.n_batch = data->params.n_batch;
    ctx_params.n_ubatch = data->params.n_ubatch;
    ctx_params.n_threads = data->params.cpuparams.n_threads;
    ctx_params.n_threads_batch = data->params.cpuparams_batch.n_threads;
    ctx_params.embeddings = embeddings_mode;
    ctx_params.flash_attn = flash_attn;
    ctx_params.no_perf = no_perf;
    ctx_params.type_k = (ggml_type)cache_type_k;
    ctx_params.type_v = (ggml_type)cache_type_v;
    ctx_params.offload_kqv = !no_kv_offload;
    ctx_params.op_offload = !no_op_offload;
    ctx_params.swa_full = swa_full;

    // RoPE configuration
    ctx_params.rope_scaling_type = (llama_rope_scaling_type)rope_scaling_type;
    if (rope_freq_base > 0) ctx_params.rope_freq_base = rope_freq_base;
    if (rope_freq_scale > 0) ctx_params.rope_freq_scale = rope_freq_scale;

    // YaRN configuration
    if (yarn_ext_factor >= 0) ctx_params.yarn_ext_factor = yarn_ext_factor;
    ctx_params.yarn_attn_factor = yarn_attn_factor;
    ctx_params.yarn_beta_slow = yarn_beta_slow;
    ctx_params.yarn_beta_fast = yarn_beta_fast;
    if (yarn_orig_ctx > 0) ctx_params.yarn_orig_ctx = yarn_orig_ctx;

    // Defragmentation threshold
    if (defrag_threshold >= 0) {
        ctx_params.defrag_thold = defrag_threshold;
    }

    data->ctx = llama_init_from_model(data->model, ctx_params);
    if (!data->ctx) {
        std::cerr << "Failed to create context" << std::endl;
        set_last_error("Failed to create context from model");
        llama_model_free(data->model);
        return 0;
    }

    // Apply NUMA configuration if requested
    if (numa_mode > 0) {
        ggml_numa_strategy numa_strategy = GGML_NUMA_STRATEGY_DISABLED;
        switch (numa_mode) {
            case 1: numa_strategy = GGML_NUMA_STRATEGY_DISTRIBUTE; break;
            case 2: numa_strategy = GGML_NUMA_STRATEGY_ISOLATE; break;
            case 3: numa_strategy = GGML_NUMA_STRATEGY_NUMACTL; break;
        }
        llama_numa_init(numa_strategy);
    }

    // Initialize sampler - will be recreated in generate with actual parameters
    data->sampler = nullptr;

    // Initialize formatted_chat buffer with context size
    data->formatted_chat.resize(data->n_ctx * 4); // Extra space for formatting

    LlamaContextData* released_data = data.release();
    {
        std::lock_guard<std::mutex> lock(g_context_registry_mutex);
        g_active_contexts.insert(released_data);
    }

    return reinterpret_cast<jlong>(released_data);
}

JNIEXPORT void JNICALL Java_org_llm_wrapper_LlamaCpp_llama_1destroy
  (JNIEnv *, jclass, jlong handle) {
    LlamaContextData* data = reinterpret_cast<LlamaContextData*>(handle);
    if (!is_valid_context(data)) return;

    {
        std::lock_guard<std::mutex> lock(g_context_registry_mutex);
        g_active_contexts.erase(data);
    }

    // Free chat messages using helper function
    free_chat_messages(data->chat_messages);
    if (data->sampler) llama_sampler_free(data->sampler);
    if (data->ctx) llama_free(data->ctx);
    if (data->model) llama_model_free(data->model);
    delete data;
}

JNIEXPORT jint JNICALL Java_org_llm_wrapper_LlamaCpp_llama_1generate
  (JNIEnv *env, jclass, jlong handle, jobject generateParams) {

    LlamaContextData* data = reinterpret_cast<LlamaContextData*>(handle);
    if (!is_valid_context(data)) {
        set_last_error("Invalid context handle");
        return -1;
    }
    if (!data || !data->ctx) {
        set_last_error("Context is null or not initialized");
        return -1;
    }

    // Extract generation parameters
    jstring prompt_jstr = getStringField(env, generateParams, "prompt");
    const char* prompt_cstr = nullptr;
    if (prompt_jstr) {
        prompt_cstr = env->GetStringUTFChars(prompt_jstr, nullptr);
    }
    if (!prompt_cstr) return -1;

    std::string user_input(prompt_cstr);

    // Extract ALL parameters from GenerateParams

    // === Basic Parameters ===
    int n_predict = getIntField(env, generateParams, "nPredict");
    bool escape = getBooleanField(env, generateParams, "escape");
    bool no_escape = getBooleanField(env, generateParams, "noEscape");
    bool verbose = getBooleanField(env, generateParams, "verbose");

    // Apply escape sequence processing
    // Process escapes by default unless no_escape is explicitly set
    if (!no_escape && escape) {
        string_process_escapes(user_input);
    }

    // === Basic Sampling Parameters ===
    float temp = getFloatField(env, generateParams, "temp");
    int top_k = getIntField(env, generateParams, "topK");
    float top_p = getFloatField(env, generateParams, "topP");
    float min_p = getFloatField(env, generateParams, "minP");
    float tfs_z = getFloatField(env, generateParams, "tfsZ");
    float typical_p = getFloatField(env, generateParams, "typicalP");
    float top_nsigma = getFloatField(env, generateParams, "topNsigma");
    int seed = getIntField(env, generateParams, "seed");

    // === Repetition Penalty Parameters ===
    float repeat_penalty = getFloatField(env, generateParams, "repeatPenalty");
    int repeat_last_n = getIntField(env, generateParams, "repeatLastN");
    float frequency_penalty = getFloatField(env, generateParams, "frequencyPenalty");
    float presence_penalty = getFloatField(env, generateParams, "presencePenalty");
    bool penalize_nl = getBooleanField(env, generateParams, "penalizeNl");
    // penalize_nl is now implemented via logit bias

    // === Mirostat Parameters ===
    int mirostat = getIntField(env, generateParams, "mirostat");
    float mirostat_tau = getFloatField(env, generateParams, "mirostatTau");
    float mirostat_eta = getFloatField(env, generateParams, "mirostatEta");

    // === DRY Parameters ===
    float dry_multiplier = getFloatField(env, generateParams, "dryMultiplier");
    float dry_base = getFloatField(env, generateParams, "dryBase");
    int dry_allowed_length = getIntField(env, generateParams, "dryAllowedLength");
    int dry_penalty_last_n = getIntField(env, generateParams, "dryPenaltyLastN");

    // Custom DRY sequence breakers
    jobjectArray dry_breakers_array = nullptr;
    {
        jclass cls = env->GetObjectClass(generateParams);
        if (cls) {
            jfieldID fid = env->GetFieldID(cls, "drySequenceBreakers", "[Ljava/lang/String;");
            if (fid) {
                dry_breakers_array = (jobjectArray)env->GetObjectField(generateParams, fid);
            } else if (env->ExceptionCheck()) {
                env->ExceptionDescribe();
                env->ExceptionClear();
                fprintf(stderr, "WARNING: Field drySequenceBreakers not found\n");
            }
        }
    }
    std::vector<std::string> dry_breakers_vec;
    if (dry_breakers_array) {
        jsize breaker_count = env->GetArrayLength(dry_breakers_array);
        for (jsize i = 0; i < breaker_count; i++) {
            jstring breaker = (jstring)env->GetObjectArrayElement(dry_breakers_array, i);
            if (breaker) {
                const char* breaker_cstr = env->GetStringUTFChars(breaker, nullptr);
                dry_breakers_vec.push_back(breaker_cstr);
                env->ReleaseStringUTFChars(breaker, breaker_cstr);
            }
        }
    }

    // === XTC Parameters ===
    float xtc_threshold = getFloatField(env, generateParams, "xtcThreshold");
    float xtc_probability = getFloatField(env, generateParams, "xtcProbability");
    int xtc_min = getIntField(env, generateParams, "xtcMin");
    (void)xtc_min; // TODO: Apply when XTC sampler is available

    // === Dynamic Temperature ===
    float dyn_temp_range = getFloatField(env, generateParams, "dynTempRange");
    float dyn_temp_exponent = getFloatField(env, generateParams, "dynTempExponent");

    // === Grammar and Constraints ===
    jstring grammar_str = getStringField(env, generateParams, "grammar");
    std::string grammar;
    if (grammar_str) {
        const char* grammar_cstr = env->GetStringUTFChars(grammar_str, nullptr);
        grammar = grammar_cstr;
        env->ReleaseStringUTFChars(grammar_str, grammar_cstr);
    }

    // Load grammar from file if specified
    jstring grammar_file_str = getStringField(env, generateParams, "grammarFile");
    if (grammar_file_str && grammar.empty()) {
        const char* grammar_file_cstr = env->GetStringUTFChars(grammar_file_str, nullptr);
        if (grammar_file_cstr && strlen(grammar_file_cstr) > 0) {
            std::ifstream grammar_file(grammar_file_cstr);
            if (grammar_file.is_open()) {
                std::stringstream buffer;
                buffer << grammar_file.rdbuf();
                grammar = buffer.str();
                if (verbose) {
                    fprintf(stderr, "Loaded grammar from file: %s\n", grammar_file_cstr);
                }
            } else {
                fprintf(stderr, "Warning: Could not open grammar file: %s\n", grammar_file_cstr);
            }
        }
        env->ReleaseStringUTFChars(grammar_file_str, grammar_file_cstr);
    }

    // JSON Schema support
    jstring json_schema_str = getStringField(env, generateParams, "jsonSchema");
    if (json_schema_str && grammar.empty()) {
        const char* schema_cstr = env->GetStringUTFChars(json_schema_str, nullptr);
        if (schema_cstr && strlen(schema_cstr) > 0) {
            try {
                auto schema = nlohmann::ordered_json::parse(schema_cstr);
                grammar = json_schema_to_grammar(schema, "");
                if (verbose) {
                    fprintf(stderr, "Converted JSON schema to grammar\n");
                }
            } catch (const std::exception& e) {
                fprintf(stderr, "Warning: Failed to parse JSON schema: %s\n", e.what());
            }
        }
        env->ReleaseStringUTFChars(json_schema_str, schema_cstr);
    }

    // Load JSON Schema from file
    jstring json_schema_file_str = getStringField(env, generateParams, "jsonSchemaFile");
    if (json_schema_file_str && grammar.empty()) {
        const char* schema_file_cstr = env->GetStringUTFChars(json_schema_file_str, nullptr);
        if (schema_file_cstr && strlen(schema_file_cstr) > 0) {
            std::ifstream schema_file(schema_file_cstr);
            if (schema_file.is_open()) {
                try {
                    nlohmann::ordered_json schema;
                    schema_file >> schema;
                    grammar = json_schema_to_grammar(schema, "");
                    if (verbose) {
                        fprintf(stderr, "Loaded and converted JSON schema from file: %s\n", schema_file_cstr);
                    }
                } catch (const std::exception& e) {
                    fprintf(stderr, "Warning: Failed to parse JSON schema file: %s\n", e.what());
                }
            } else {
                fprintf(stderr, "Warning: Could not open JSON schema file: %s\n", schema_file_cstr);
            }
        }
        env->ReleaseStringUTFChars(json_schema_file_str, schema_file_cstr);
    }

    // === Token Control ===
    int keep = getIntField(env, generateParams, "keep");
    bool ignore_eos = getBooleanField(env, generateParams, "ignoreEos");
    bool no_context_shift = getBooleanField(env, generateParams, "noContextShift");
    bool no_penalize_eos = getBooleanField(env, generateParams, "noPenalizeEos");
    // no_context_shift is applied during generation
    // no_penalize_eos is implemented via logit bias compensation

    // === Logit Bias ===
    jstring logit_bias_str = getStringField(env, generateParams, "logitBias");
    std::vector<llama_logit_bias> custom_logit_biases;
    if (logit_bias_str) {
        const char* bias_cstr = env->GetStringUTFChars(logit_bias_str, nullptr);
        if (bias_cstr && strlen(bias_cstr) > 0) {
            // Parse logit bias string format: "TOKEN_ID(+/-)BIAS,TOKEN_ID(+/-)BIAS,..."
            std::string bias_str(bias_cstr);
            std::stringstream ss(bias_str);
            std::string pair;

            while (std::getline(ss, pair, ',')) {
                // Trim whitespace
                pair.erase(0, pair.find_first_not_of(" \t"));
                pair.erase(pair.find_last_not_of(" \t") + 1);

                if (!pair.empty()) {
                    // Find the bias value (starts with + or -, or just a number)
                    size_t bias_pos = pair.find_last_of("+-");
                    if (bias_pos == std::string::npos) {
                        // No explicit sign, look for space or other separator
                        bias_pos = pair.find_last_of(" ");
                    } else if (bias_pos > 0 && std::isdigit(pair[bias_pos - 1])) {
                        // The +/- is part of a number like "123-45", find the actual separator
                        size_t temp_pos = pair.find_last_of(" ", bias_pos);
                        if (temp_pos != std::string::npos) {
                            bias_pos = temp_pos;
                        }
                    }

                    if (bias_pos != std::string::npos && bias_pos > 0) {
                        try {
                            int token_id = std::stoi(pair.substr(0, bias_pos));
                            float bias = std::stof(pair.substr(bias_pos));
                            custom_logit_biases.push_back({token_id, bias});

                            if (verbose) {
                                fprintf(stderr, "Added logit bias: token %d, bias %.2f\n", token_id, bias);
                            }
                        } catch (...) {
                            fprintf(stderr, "Warning: Invalid logit bias pair: %s\n", pair.c_str());
                        }
                    }
                }
            }
        }
        env->ReleaseStringUTFChars(logit_bias_str, bias_cstr);
    }

    // === Output Control ===
    int n_probs = getIntField(env, generateParams, "nProbs");
    int min_keep = getIntField(env, generateParams, "minKeep");
    // n_probs will be used to return token probabilities after sampling

    // === Group Attention ===
    int grp_attn_n = getIntField(env, generateParams, "grpAttnN");
    int grp_attn_w = getIntField(env, generateParams, "grpAttnW");
    (void)grp_attn_n; // TODO: Apply group attention parameters
    (void)grp_attn_w; // TODO: Apply group attention width

    // === Conversation Mode ===
    bool conversation = getBooleanField(env, generateParams, "conversation");
    bool chat_template = getBooleanField(env, generateParams, "chatTemplate");
    bool multiline = getBooleanField(env, generateParams, "multiline");
    bool use_conversation = getBooleanField(env, generateParams, "useConversation");
    bool add_assistant = getBooleanField(env, generateParams, "addAssistant");
    // conversation flag enables conversation features even without chat_mode
    // chat_template controls whether to use chat template formatting
    // multiline is handled at the UI level - the library receives complete strings
    (void)multiline; // UI concern - input already processed
    (void)use_conversation; // Legacy - covered by conversation flag

    // === Output Formatting ===
    bool no_display_prompt = getBooleanField(env, generateParams, "noDisplayPrompt");
    bool no_parse_special_tokens = getBooleanField(env, generateParams, "noParseSpecialTokens");
    bool display_prompt = getBooleanField(env, generateParams, "displayPrompt");
    // These will be used for output formatting during token generation
    // display_prompt and no_display_prompt are now used for prompt echo
    // no_parse_special_tokens is now used in llama_token_to_piece calls

    // === Special Tokens ===
    jstring bos_token_str = getStringField(env, generateParams, "bosToken");
    std::string bos_token;
    if (bos_token_str) {
        const char* bos_cstr = env->GetStringUTFChars(bos_token_str, nullptr);
        bos_token = bos_cstr;
        env->ReleaseStringUTFChars(bos_token_str, bos_cstr);
    }

    jstring eos_token_str = getStringField(env, generateParams, "eosToken");
    std::string eos_token;
    if (eos_token_str) {
        const char* eos_cstr = env->GetStringUTFChars(eos_token_str, nullptr);
        eos_token = eos_cstr;
        env->ReleaseStringUTFChars(eos_token_str, eos_cstr);
    }

    // === Infill Mode ===
    bool infill = getBooleanField(env, generateParams, "infill");
    jstring input_prefix_str = getStringField(env, generateParams, "inputPrefix");
    std::string input_prefix;
    if (input_prefix_str) {
        const char* prefix_cstr = env->GetStringUTFChars(input_prefix_str, nullptr);
        input_prefix = prefix_cstr;
        env->ReleaseStringUTFChars(input_prefix_str, prefix_cstr);
    }

    jstring input_suffix_str = getStringField(env, generateParams, "inputSuffix");
    std::string input_suffix;
    if (input_suffix_str) {
        const char* suffix_cstr = env->GetStringUTFChars(input_suffix_str, nullptr);
        input_suffix = suffix_cstr;
        env->ReleaseStringUTFChars(input_suffix_str, suffix_cstr);
    }

    // === Cache Control ===
    bool cache_prompt = getBooleanField(env, generateParams, "cachePrompt");
    jstring cache_session_str = getStringField(env, generateParams, "cacheSession");
    std::string cache_session;
    if (cache_session_str) {
        const char* session_cstr = env->GetStringUTFChars(cache_session_str, nullptr);
        cache_session = session_cstr;
        env->ReleaseStringUTFChars(cache_session_str, session_cstr);
    }
    bool truncate_prompt = getBooleanField(env, generateParams, "truncatePrompt");
    // cache_prompt controls whether to cache prompt tokens
    // cache_session will be used for session save/load when implemented
    // truncate_prompt is applied after tokenization

    // === CFG (Classifier-Free Guidance) ===
    float guidance_scale = getFloatField(env, generateParams, "guidanceScale");
    jstring negative_prompt_str = getStringField(env, generateParams, "negativePrompt");
    std::string negative_prompt;
    if (negative_prompt_str) {
        const char* neg_cstr = env->GetStringUTFChars(negative_prompt_str, nullptr);
        negative_prompt = neg_cstr;
        env->ReleaseStringUTFChars(negative_prompt_str, neg_cstr);
    }
    // TODO: Implement CFG when API supports it
    (void)guidance_scale;
    (void)negative_prompt;

    // === Speculative Decoding ===
    int n_draft = getIntField(env, generateParams, "nDraft");
    float p_accept = getFloatField(env, generateParams, "pAccept");
    float p_split = getFloatField(env, generateParams, "pSplit");
    // TODO: Implement speculative decoding when API supports it
    (void)n_draft;
    (void)p_accept;
    (void)p_split;

    // === Performance ===
    bool no_warmup = getBooleanField(env, generateParams, "noWarmup");
    // TODO: Skip warmup when this is true
    (void)no_warmup;

    // === Per-Generation Thread/Batch Control ===
    int n_threads_gen = getIntField(env, generateParams, "nThreads");
    int n_batch_gen = getIntField(env, generateParams, "nBatch");

    // Apply thread settings for this generation if specified
    if (n_threads_gen > 0) {
        data->params.cpuparams.n_threads = n_threads_gen;
        if (verbose) {
            fprintf(stderr, "Using %d threads for this generation\n", n_threads_gen);
        }
    }

    // Note: n_batch cannot be changed per-generation as it affects memory allocation
    // It's a context-level parameter that must be set during initialization
    if (n_batch_gen > 0 && verbose) {
        fprintf(stderr, "WARNING: n_batch cannot be changed per-generation. Use InitParams.nBatch instead.\n");
    }

    // === Custom Sampler Configuration ===
    jstring samplers_str = getStringField(env, generateParams, "samplers");
    std::vector<std::string> custom_samplers;
    if (samplers_str) {
        const char* samplers_cstr = env->GetStringUTFChars(samplers_str, nullptr);
        if (samplers_cstr && strlen(samplers_cstr) > 0) {
            // Parse comma-separated sampler names
            std::stringstream ss(samplers_cstr);
            std::string sampler_name;
            while (std::getline(ss, sampler_name, ',')) {
                // Trim whitespace
                sampler_name.erase(0, sampler_name.find_first_not_of(" \t"));
                sampler_name.erase(sampler_name.find_last_not_of(" \t") + 1);
                if (!sampler_name.empty()) {
                    custom_samplers.push_back(sampler_name);
                }
            }
            if (verbose) {
                fprintf(stderr, "Custom sampler order: ");
                for (const auto& s : custom_samplers) {
                    fprintf(stderr, "%s ", s.c_str());
                }
                fprintf(stderr, "\n");
            }
        }
        env->ReleaseStringUTFChars(samplers_str, samplers_cstr);
    }

    // Alternative: samplingSeq for detailed sampling configuration
    jstring sampling_seq_str = getStringField(env, generateParams, "samplingSeq");
    if (sampling_seq_str && custom_samplers.empty()) {
        const char* seq_cstr = env->GetStringUTFChars(sampling_seq_str, nullptr);
        if (seq_cstr && strlen(seq_cstr) > 0) {
            // Parse sampling sequence (format: "top_k:40;top_p:0.9;temp:0.8")
            std::string seq_str(seq_cstr);
            if (verbose) {
                fprintf(stderr, "Sampling sequence: %s\n", seq_str.c_str());
            }
            // TODO: Parse and apply custom sampling sequence when API supports it
        }
        env->ReleaseStringUTFChars(sampling_seq_str, seq_cstr);
    }

    // === XTC Probabilistic Threshold ===
    bool xtc_probabilistic_threshold = getBooleanField(env, generateParams, "xtcProbabilisticThreshold");
    // TODO: Apply when XTC sampler supports probabilistic threshold mode
    (void)xtc_probabilistic_threshold;

    // === Plain Text Mode ===
    bool chat_mode = getBooleanField(env, generateParams, "chatMode");

    // === Additional missing fields ===
    // Stop sequences
    jobjectArray stopSequences = nullptr;
    {
        jclass cls = env->GetObjectClass(generateParams);
        if (cls) {
            jfieldID fid = env->GetFieldID(cls, "stopSequences", "[Ljava/lang/String;");
            if (fid) {
                stopSequences = (jobjectArray)env->GetObjectField(generateParams, fid);
            } else if (env->ExceptionCheck()) {
                env->ExceptionDescribe();
                env->ExceptionClear();
                fprintf(stderr, "WARNING: Field stopSequences not found\n");
            }
        }
    }
    std::vector<std::string> stop_sequences;
    if (stopSequences) {
        jsize stop_count = env->GetArrayLength(stopSequences);
        for (jsize i = 0; i < stop_count; i++) {
            jstring stop_str = (jstring)env->GetObjectArrayElement(stopSequences, i);
            if (stop_str) {
                const char* stop_cstr = env->GetStringUTFChars(stop_str, nullptr);
                stop_sequences.push_back(stop_cstr);
                env->ReleaseStringUTFChars(stop_str, stop_cstr);
            }
        }
    }

    // BOS token control
    bool add_bos = getBooleanField(env, generateParams, "addBos");

    // File input
    jstring file_str = getStringField(env, generateParams, "file");
    if (file_str) {
        const char* file_cstr = env->GetStringUTFChars(file_str, nullptr);
        if (strlen(file_cstr) > 0) {
            std::ifstream file(file_cstr);
            if (file.is_open()) {
                std::stringstream buffer;
                buffer << file.rdbuf();
                user_input = buffer.str();
            }
        }
        env->ReleaseStringUTFChars(file_str, file_cstr);
    }

    // Binary file input
    jstring binary_file_str = getStringField(env, generateParams, "binaryFile");
    if (binary_file_str) {
        const char* binary_file_cstr = env->GetStringUTFChars(binary_file_str, nullptr);
        if (strlen(binary_file_cstr) > 0) {
            std::ifstream file(binary_file_cstr, std::ios::binary);
            if (file.is_open()) {
                // Read binary file as-is
                std::vector<char> buffer((std::istreambuf_iterator<char>(file)),
                                         std::istreambuf_iterator<char>());
                user_input = std::string(buffer.begin(), buffer.end());
                if (verbose) {
                    fprintf(stderr, "Loaded binary file: %s (%zu bytes)\n",
                            binary_file_cstr, user_input.size());
                }
            } else {
                fprintf(stderr, "Warning: Could not open binary file: %s\n", binary_file_cstr);
            }
        }
        env->ReleaseStringUTFChars(binary_file_str, binary_file_cstr);
    }

    // Get vocab from model
    const struct llama_vocab * vocab = llama_model_get_vocab(data->model);

    // Get default tokens for reference
    llama_token default_bos = llama_vocab_bos(vocab);
    llama_token default_eos = llama_vocab_eos(vocab);
    llama_token custom_bos = default_bos;
    llama_token custom_eos = default_eos;

    // Override BOS/EOS tokens if custom ones provided
    if (!bos_token.empty()) {
        // Tokenize the custom BOS string to get its token ID
        std::vector<llama_token> bos_tokens;
        int n_bos = llama_tokenize(vocab, bos_token.c_str(), bos_token.size(),
                                  nullptr, 0, false, true);
        if (n_bos > 0) {
            bos_tokens.resize(n_bos);
            llama_tokenize(vocab, bos_token.c_str(), bos_token.size(),
                          bos_tokens.data(), bos_tokens.size(), false, true);
            if (!bos_tokens.empty()) {
                custom_bos = bos_tokens[0];
                if (verbose) {
                    fprintf(stderr, "Using custom BOS token: %d (was %d)\n",
                            custom_bos, default_bos);
                }
            }
        }
    }

    if (!eos_token.empty()) {
        // Tokenize the custom EOS string to get its token ID
        std::vector<llama_token> eos_tokens;
        int n_eos = llama_tokenize(vocab, eos_token.c_str(), eos_token.size(),
                                  nullptr, 0, false, true);
        if (n_eos > 0) {
            eos_tokens.resize(n_eos);
            llama_tokenize(vocab, eos_token.c_str(), eos_token.size(),
                          eos_tokens.data(), eos_tokens.size(), false, true);
            if (!eos_tokens.empty()) {
                custom_eos = eos_tokens[0];
                if (verbose) {
                    fprintf(stderr, "Using custom EOS token: %d (was %d)\n",
                            custom_eos, default_eos);
                }
            }
        }
    }

    // Get keep parameter from GenerateParams
    if (keep != -1 && keep != data->n_keep) {
        data->n_keep = keep;
    }

    // Debug output
    if (verbose) {
        fprintf(stderr, "\n=== Generate parameters ===\n");
        fprintf(stderr, "n_predict: %d\n", n_predict);
        fprintf(stderr, "temp: %.2f\n", temp);
        fprintf(stderr, "top_k: %d\n", top_k);
        fprintf(stderr, "top_p: %.2f\n", top_p);
        fprintf(stderr, "min_p: %.2f\n", min_p);
        fprintf(stderr, "tfs_z: %.2f\n", tfs_z);
        fprintf(stderr, "typical_p: %.2f\n", typical_p);
        fprintf(stderr, "repeat_penalty: %.2f\n", repeat_penalty);
        fprintf(stderr, "repeat_last_n: %d\n", repeat_last_n);
        fprintf(stderr, "frequency_penalty: %.2f\n", frequency_penalty);
        fprintf(stderr, "presence_penalty: %.2f\n", presence_penalty);
        fprintf(stderr, "mirostat: %d\n", mirostat);
        fprintf(stderr, "seed: %d\n", seed);
        fprintf(stderr, "n_keep: %d\n", data->n_keep);
        fprintf(stderr, "=== End parameters ===\n\n");
    }

    // Free old sampler and create new one with current parameters
    if (data->sampler) {
        llama_sampler_free(data->sampler);
    }

    // Create sampler chain with ALL parameters
    data->sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());

    // Add DRY sampler if enabled
    if (dry_multiplier > 0.0f) {
        std::vector<const char*> dry_breakers;

        // Use custom breakers if provided, otherwise use defaults
        if (!dry_breakers_vec.empty()) {
            for (const auto& breaker : dry_breakers_vec) {
                dry_breakers.push_back(breaker.c_str());
            }
        } else {
            // Default breakers
            dry_breakers = {"\n", "###", "\\n", ":", "\"", "*"};
        }

        // Get n_ctx_train from model
        int32_t n_ctx_train = llama_model_n_ctx_train(data->model);

        llama_sampler_chain_add(data->sampler,
            llama_sampler_init_dry(vocab, n_ctx_train, dry_multiplier, dry_base,
                                 dry_allowed_length, dry_penalty_last_n,
                                 dry_breakers.data(), dry_breakers.size()));
    }

    // Add samplers in the correct order (matching llama-cli)

    // Create logit bias for penalize_nl if needed
    std::vector<llama_logit_bias> logit_biases;
    if (penalize_nl) {
        // Get newline token ID and add negative bias
        llama_token nl_token = llama_vocab_nl(vocab);
        if (nl_token >= 0) {
            logit_biases.push_back({nl_token, -5.0f}); // Strong penalty for newline
        }
    }

    // Add positive bias to EOS tokens if no_penalize_eos is set
    if (no_penalize_eos && (repeat_penalty != 1.0f || frequency_penalty != 0.0f || presence_penalty != 0.0f)) {
        // Get all EOS tokens and add positive bias to compensate for penalties
        std::vector<llama_token> eos_tokens;

        // Add default EOS token
        llama_token default_eos_token = llama_vocab_eos(vocab);
        if (default_eos_token >= 0) {
            // Calculate compensation bias based on penalty strength
            float compensation = 0.0f;
            if (repeat_penalty > 1.0f) {
                compensation += logf(repeat_penalty) * 2.0f; // Compensate for repeat penalty
            }
            if (frequency_penalty > 0.0f) {
                compensation += frequency_penalty * 2.0f; // Compensate for frequency penalty
            }
            if (presence_penalty > 0.0f) {
                compensation += presence_penalty; // Compensate for presence penalty
            }

            if (compensation > 0.0f) {
                logit_biases.push_back({default_eos_token, compensation});
                if (verbose) {
                    fprintf(stderr, "Adding EOS compensation bias: %.2f for token %d\n",
                            compensation, default_eos_token);
                }
            }
        }

        // Also handle custom EOS if different from default
        if (custom_eos != default_eos && custom_eos >= 0) {
            float compensation = 0.0f;
            if (repeat_penalty > 1.0f) {
                compensation += logf(repeat_penalty) * 2.0f;
            }
            if (frequency_penalty > 0.0f) {
                compensation += frequency_penalty * 2.0f;
            }
            if (presence_penalty > 0.0f) {
                compensation += presence_penalty;
            }

            if (compensation > 0.0f) {
                logit_biases.push_back({custom_eos, compensation});
                if (verbose) {
                    fprintf(stderr, "Adding custom EOS compensation bias: %.2f for token %d\n",
                            compensation, custom_eos);
                }
            }
        }
    }

    // Add custom logit biases from logitBias parameter
    for (const auto& bias : custom_logit_biases) {
        logit_biases.push_back(bias);
    }

    // Add logit bias sampler
    if (!logit_biases.empty()) {
        llama_sampler_chain_add(data->sampler,
            llama_sampler_init_logit_bias(llama_vocab_n_tokens(vocab),
                                         logit_biases.size(),
                                         logit_biases.data()));
    } else {
        // Empty logit bias sampler
        llama_sampler_chain_add(data->sampler,
            llama_sampler_init_logit_bias(llama_vocab_n_tokens(vocab), 0, nullptr));
    }

    // Add penalties sampler if any penalty is set
    if (repeat_penalty != 1.0f || frequency_penalty != 0.0f || presence_penalty != 0.0f) {
        // no_penalize_eos is handled via logit bias compensation above

        llama_sampler_chain_add(data->sampler,
            llama_sampler_init_penalties(
                repeat_last_n,
                repeat_penalty,
                frequency_penalty,
                presence_penalty
            ));
    }

    // Top-n-sigma sampling
    if (top_nsigma > 0.0f) {
        llama_sampler_chain_add(data->sampler, llama_sampler_init_top_n_sigma(top_nsigma));
    }

    // Top-K sampling
    if (top_k > 0) {
        llama_sampler_chain_add(data->sampler, llama_sampler_init_top_k(top_k));
    }

    // Tail-free sampling
    if (tfs_z < 1.0f) {
        // TODO: Tail-free sampling has been removed from llama.cpp API
        // It may have been integrated into another sampler or deprecated
        // For now, we skip TFS when tfs_z < 1.0
        if (verbose) {
            fprintf(stderr, "WARNING: Tail-free sampling (tfs_z=%.2f) requested but not available in current API\n", tfs_z);
        }
    }

    // Typical sampling
    if (typical_p < 1.0f) {
        llama_sampler_chain_add(data->sampler, llama_sampler_init_typical(typical_p, 1));
    }

    // Top-P sampling
    if (top_p < 1.0f) {
        llama_sampler_chain_add(data->sampler, llama_sampler_init_top_p(top_p, min_keep > 0 ? min_keep : 1));
    }

    // Min-P sampling
    if (min_p > 0.0f) {
        llama_sampler_chain_add(data->sampler, llama_sampler_init_min_p(min_p, min_keep > 0 ? min_keep : 1));
    }

    // XTC sampling
    if (xtc_probability > 0.0f && xtc_threshold > 0.0f) {
        llama_sampler_chain_add(data->sampler,
            llama_sampler_init_xtc(xtc_probability, xtc_threshold, xtc_min, seed));
    }

    // Dynamic temperature or regular temperature
    if (dyn_temp_range > 0.0f) {
        llama_sampler_chain_add(data->sampler,
            llama_sampler_init_temp_ext(temp, dyn_temp_range, dyn_temp_exponent));
    } else {
        llama_sampler_chain_add(data->sampler, llama_sampler_init_temp(temp));
    }

    // Mirostat sampling (replaces temperature)
    if (mirostat == 1) {
        llama_sampler_chain_add(data->sampler,
            llama_sampler_init_mirostat(llama_vocab_n_tokens(vocab), seed, mirostat_tau, mirostat_eta, 100));
    } else if (mirostat == 2) {
        llama_sampler_chain_add(data->sampler,
            llama_sampler_init_mirostat_v2(seed, mirostat_tau, mirostat_eta));
    } else {
        // Regular sampling
        uint32_t actual_seed = (seed == -1) ? LLAMA_DEFAULT_SEED : (uint32_t)seed;
        llama_sampler_chain_add(data->sampler, llama_sampler_init_dist(actual_seed));
    }

    // Apply grammar constraint if provided (must be last in the chain)
    if (!grammar.empty()) {
        // Parse and create grammar sampler
        llama_sampler * grammar_sampler = llama_sampler_init_grammar(vocab, grammar.c_str(), "root");
        if (grammar_sampler) {
            llama_sampler_chain_add(data->sampler, grammar_sampler);
            if (verbose) {
                fprintf(stderr, "Applied grammar constraint\n");
            }
        } else {
            fprintf(stderr, "WARNING: Failed to create grammar sampler\n");
        }
    }

    std::string complete_prompt;

    // Handle infill mode
    if (infill && !input_prefix.empty()) {
        // In infill mode, construct prompt as: prefix + user_input + suffix
        complete_prompt = input_prefix + user_input;
        if (!input_suffix.empty()) {
            complete_prompt += input_suffix;
        }

        if (verbose) {
            fprintf(stderr, "Infill mode - prefix: %s, suffix: %s\n",
                    input_prefix.c_str(), input_suffix.c_str());
        }
    } else if (chat_mode || conversation) {
        // Chat mode or conversation mode - use history and optionally templates
        // Get chat template if enabled
        const char * tmpl = (chat_template != false) ? llama_model_chat_template(data->model, nullptr) : nullptr;

        // Add user message to chat history
        data->chat_messages.push_back({"user", strdup(user_input.c_str())});

        // Format ALL messages with chat template to get complete prompt
        if (data->formatted_chat.empty()) {
            data->formatted_chat.resize(data->n_ctx * 4);
        }

        int new_len;
        if (tmpl == nullptr) {
            // No template found - use simple format
            fprintf(stderr, "No chat template found, using simple format\n");

            // Build complete conversation
            for (size_t i = 0; i < data->chat_messages.size(); i++) {
                if (i > 0) complete_prompt += "\n";
                complete_prompt += data->chat_messages[i].content;
            }

            new_len = complete_prompt.length();
            if (new_len > (int)data->formatted_chat.size()) {
                data->formatted_chat.resize(new_len * 2);
                new_len = complete_prompt.length();
            }
            memcpy(data->formatted_chat.data(), complete_prompt.c_str(), new_len);
        } else {
            // Use the model's chat template
            new_len = llama_chat_apply_template(tmpl, data->chat_messages.data(),
                                               data->chat_messages.size(), true,
                                               data->formatted_chat.data(), data->formatted_chat.size());

            if (new_len > (int)data->formatted_chat.size()) {
                data->formatted_chat.resize(new_len * 2);
                new_len = llama_chat_apply_template(tmpl, data->chat_messages.data(),
                                                  data->chat_messages.size(), true,
                                                  data->formatted_chat.data(), data->formatted_chat.size());
            }
        }

        if (new_len < 0) {
            env->ReleaseStringUTFChars(prompt_jstr, prompt_cstr);
            set_last_error("Failed to apply chat template");
            return -1;
        }

        // Get the COMPLETE prompt from formatted chat
        complete_prompt = std::string(data->formatted_chat.begin(), data->formatted_chat.begin() + new_len);
    } else {
        // Plain text mode - use prompt as is, no history, no templates
        complete_prompt = user_input;
        fprintf(stderr, "Using plain text mode\n");
    }

    env->ReleaseStringUTFChars(prompt_jstr, prompt_cstr);

    // Session loading - restore state if session file exists
    if (!cache_session.empty()) {
        // Check if session file exists
        std::ifstream session_file(cache_session);
        if (session_file.good()) {
            session_file.close();

            if (verbose) {
                fprintf(stderr, "\n=== Loading session from: %s ===\n", cache_session.c_str());
            }

            // Load session tokens
            std::vector<llama_token> loaded_tokens(data->n_ctx);
            size_t n_loaded = 0;

            bool load_success = llama_state_load_file(data->ctx, cache_session.c_str(),
                                                     loaded_tokens.data(), loaded_tokens.size(),
                                                     &n_loaded);

            if (load_success && n_loaded > 0) {
                // Update our token tracking
                data->all_tokens.clear();
                data->all_tokens.insert(data->all_tokens.begin(),
                                      loaded_tokens.begin(),
                                      loaded_tokens.begin() + n_loaded);
                data->n_past = n_loaded;

                if (verbose) {
                    fprintf(stderr, "Session loaded successfully (%zu tokens)\n", n_loaded);
                    fprintf(stderr, "Continuing from loaded state\n");
                }

                // If the prompt matches the beginning of loaded tokens, skip tokenization
                // Otherwise, we'll find the common prefix normally
            } else {
                fprintf(stderr, "WARNING: Failed to load session from %s\n", cache_session.c_str());
            }
        }
    }

    // Log prompt for debugging
    fprintf(stderr, "\n=== PROMPT BEFORE TOKENIZATION ===\n");
    fprintf(stderr, "%s\n", complete_prompt.c_str());
    fprintf(stderr, "=== END PROMPT ===\n\n");

    // Check if this is a continuation of the previous prompt (for caching)
    bool is_continuation = false;
    if (!chat_mode && !data->last_prompt.empty()) {
        // In plain text mode, check if new prompt starts with the old one
        if (complete_prompt.size() > data->last_prompt.size() &&
            complete_prompt.substr(0, data->last_prompt.size()) == data->last_prompt) {
            is_continuation = true;
            fprintf(stderr, "\n=== Detected prompt continuation ===\n");
            fprintf(stderr, "Previous prompt size: %zu chars\n", data->last_prompt.size());
            fprintf(stderr, "New prompt size: %zu chars\n", complete_prompt.size());
            fprintf(stderr, "New content: %zu chars\n", complete_prompt.size() - data->last_prompt.size());
            fprintf(stderr, "===================================\n\n");
        }
    }

    // Additional cases for continuation detection:
    // 1. When using conversation mode without chat templates
    // 2. When cache_prompt is enabled and session is being reused
    // 3. When explicit continuation patterns are detected
    if (!is_continuation && cache_prompt && !data->all_tokens.empty()) {
        // If we have cached tokens and caching is enabled, it might be a continuation
        if (conversation && !chat_mode) {
            // In conversation mode without chat templates, assume continuation
            is_continuation = true;
            if (verbose) {
                fprintf(stderr, "Treating as continuation in conversation mode\n");
            }
        }
    }

    // Save current prompt for next time
    data->last_prompt = complete_prompt;

    // Tokenize the COMPLETE prompt
    // If custom BOS is provided, tokenize without BOS and add it manually
    bool use_default_bos = (custom_bos == default_bos) && add_bos;

    const int n_prompt_tokens_raw = -llama_tokenize(vocab, complete_prompt.c_str(), complete_prompt.size(),
                                              NULL, 0, use_default_bos, true);
    std::vector<llama_token> prompt_tokens(n_prompt_tokens_raw);
    if (llama_tokenize(vocab, complete_prompt.c_str(), complete_prompt.size(),
                      prompt_tokens.data(), prompt_tokens.size(), use_default_bos, true) < 0) {
        set_last_error("Failed to tokenize the prompt");
        return -1;
    }

    // Add custom BOS token if needed
    if (custom_bos != default_bos && add_bos) {
        prompt_tokens.insert(prompt_tokens.begin(), custom_bos);
        if (verbose) {
            fprintf(stderr, "Added custom BOS token %d at beginning\n", custom_bos);
        }
    }

    // Get context size for truncation check
    int n_ctx = llama_n_ctx(data->ctx);

    // Handle prompt truncation if enabled
    if (truncate_prompt && (int)prompt_tokens.size() > n_ctx - n_predict) {
        int max_prompt_tokens = n_ctx - n_predict - 1; // Leave room for generation
        if (max_prompt_tokens <= 0) {
            set_last_error("Context too small for generation with this prompt");
            return -1;
        }

        if (verbose) {
            fprintf(stderr, "Truncating prompt from %zu to %d tokens (context=%d, n_predict=%d)\n",
                    prompt_tokens.size(), max_prompt_tokens, n_ctx, n_predict);
        }

        // Keep the most recent tokens
        prompt_tokens.erase(prompt_tokens.begin(),
                           prompt_tokens.begin() + ((int)prompt_tokens.size() - max_prompt_tokens));
    }

    if (verbose) {
        fprintf(stderr, "\n=== Token processing ===\n");
        fprintf(stderr, "Original prompt tokens: %d\n", n_prompt_tokens_raw);
        fprintf(stderr, "Actual prompt tokens: %zu\n", prompt_tokens.size());
        fprintf(stderr, "Previous tokens processed: %d\n", data->n_past);
        fprintf(stderr, "n_keep: %d\n", data->n_keep);
    }

    // Get token callback
    jobject tokenCallback = getObjectField(env, generateParams, "tokenCallback", "Ljava/util/function/Predicate;");
    jclass predicateClass = nullptr;
    jmethodID testMethod = nullptr;

    if (tokenCallback) {
        predicateClass = env->GetObjectClass(tokenCallback);
        testMethod = env->GetMethodID(predicateClass, "test", "(Ljava/lang/Object;)Z");
    }

    // Find common prefix between old and new tokens
    int common_prefix = 0;
    int min_len = std::min((int)data->all_tokens.size(), (int)prompt_tokens.size());

    // If n_keep is 0 or cache_prompt is false, don't reuse any cache
    if (data->n_keep == 0 || !cache_prompt) {
        // Clear everything
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
        llama_kv_self_clear(data->ctx);
#ifdef __clang__
#pragma clang diagnostic pop
#endif
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
        data->all_tokens.clear();
        data->n_past = 0;
        common_prefix = 0;

        if (!cache_prompt && verbose) {
            fprintf(stderr, "Prompt caching disabled - cleared KV cache\n");
        }
    } else {
        // Special handling for prompt continuation
        if (is_continuation && !data->all_tokens.empty()) {
            // For continuation, the common prefix should be the size of previous tokens
            common_prefix = (int)data->all_tokens.size();

            // Verify that tokens actually match
            bool tokens_match = true;
            for (int i = 0; i < common_prefix && i < (int)prompt_tokens.size(); i++) {
                if (data->all_tokens[i] != prompt_tokens[i]) {
                    fprintf(stderr, "WARNING: Token mismatch at position %d in continuation\n", i);
                    tokens_match = false;
                    common_prefix = i;
                    break;
                }
            }

            if (tokens_match && common_prefix == data->n_past) {
                // Perfect continuation - don't touch the KV cache!
                fprintf(stderr, "Perfect continuation detected - keeping full KV cache\n");
            }
        } else {
            // Not a continuation - find common prefix normally
            for (int i = 0; i < min_len; i++) {
                if (data->all_tokens[i] == prompt_tokens[i]) {
                    common_prefix++;
                } else {
                    break;
                }
            }
        }

        // Limit by n_keep if specified
        if (data->n_keep > 0 && !is_continuation) {
            common_prefix = std::min(common_prefix, data->n_keep);
        }

        // Handle KV cache based on common prefix
        if (common_prefix < data->n_past) {
            // We have more tokens in KV cache than common prefix
            // This happens after generation - remove only the excess
            fprintf(stderr, "Adjusting KV cache: n_past=%d -> common_prefix=%d\n",
                    data->n_past, common_prefix);
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
            llama_kv_self_seq_rm(data->ctx, 0, common_prefix, data->n_past);
#ifdef __clang__
#pragma clang diagnostic pop
#endif
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
            data->n_past = common_prefix;
        }
    }

    fprintf(stderr, "Common prefix: %d tokens\n", common_prefix);
    fprintf(stderr, "Tokens to process: %zu\n", prompt_tokens.size() - common_prefix);
    fprintf(stderr, "n_past: %d, all_tokens.size: %zu\n", data->n_past, data->all_tokens.size());
    fprintf(stderr, "=== End token processing ===\n");

    // Update stored tokens based on common prefix
    if (common_prefix > 0 && !data->all_tokens.empty()) {
        // We have a common prefix - keep it and update the rest
        if (common_prefix < (int)data->all_tokens.size()) {
            // Truncate to common prefix
            data->all_tokens.resize(common_prefix);
        }
        // Append new tokens after common prefix
        if (common_prefix < (int)prompt_tokens.size()) {
            data->all_tokens.insert(data->all_tokens.end(),
                                  prompt_tokens.begin() + common_prefix,
                                  prompt_tokens.end());
        }
    } else {
        // No common prefix or no previous tokens - replace all
        data->all_tokens = prompt_tokens;
    }

    // Process new tokens in batches
    int n_batch = llama_n_batch(data->ctx);
    // n_ctx already declared above for truncation
    std::string response;

    // If display_prompt is true and we have a callback, send the prompt first
    if (display_prompt && !no_display_prompt && tokenCallback) {
        // Convert prompt tokens to text and send through callback
        for (int i = 0; i < (int)prompt_tokens.size(); i++) {
            char buf[256];
            int n = llama_token_to_piece(vocab, prompt_tokens[i], buf, sizeof(buf), 0, !no_parse_special_tokens);
            if (n > 0) {
                std::string piece(buf, n);
                jstring tokenJStr = env->NewStringUTF(piece.c_str());
                jboolean shouldContinue = env->CallBooleanMethod(tokenCallback, testMethod, tokenJStr);
                env->DeleteLocalRef(tokenJStr);

                if (!shouldContinue) {
                    return 0; // User cancelled during prompt display
                }
            }
        }
    }

    // Process only new tokens (from common_prefix to end)
    for (int i = common_prefix; i < (int)prompt_tokens.size(); i += n_batch) {
        int batch_size = std::min(n_batch, (int)prompt_tokens.size() - i);

        // Create batch with proper positions for caching
        llama_batch batch = llama_batch_init(batch_size, 0, 1);
        batch.n_tokens = batch_size;

        for (int j = 0; j < batch_size; j++) {
            batch.token[j] = prompt_tokens[i + j];
            batch.pos[j] = i + j;  // Position from start of sequence
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = 0;  // seq_id already allocated by llama_batch_init
            batch.logits[j] = 0;
        }
        batch.logits[batch_size - 1] = 1;  // Only need logits for last token in batch

        if (data->n_past + batch.n_tokens > n_ctx) {
            if (no_context_shift) {
                // Stop generation instead of shifting context
                if (verbose) {
                    fprintf(stderr, "Context full, stopping generation (no_context_shift=true)\n");
                }
                set_last_error("Context size exceeded, generation stopped");
                free(batch.seq_id[0]);
                llama_batch_free(batch);
                break;
            } else {
                // Default behavior - could implement context shifting here in future
                fprintf(stderr, "context size exceeded\n");
                set_last_error("Context size exceeded");
                free(batch.seq_id[0]);
                llama_batch_free(batch);
                break;
            }
        }

        int ret = llama_decode(data->ctx, batch);

        llama_batch_free(batch);

        if (ret != 0) {
            set_last_error("Failed to decode prompt batch, ret = " + std::to_string(ret));
            return -1;
        }

        data->n_past += batch_size;
    }

    // Now generate the response token by token
    llama_token new_token_id;
    int n_generated = 0;

    // Handle n_predict = -1 (generate until context full)
    int max_tokens = n_predict;
    if (n_predict == -1) {
        // Generate until context is full (leave 1 token space)
        max_tokens = n_ctx - data->n_past - 1;
        if (verbose) {
            fprintf(stderr, "n_predict=-1: generating up to %d tokens (context=%d, used=%d)\n",
                    max_tokens, n_ctx, data->n_past);
        }
    }

    // Structure to store token probabilities if n_probs > 0
    struct TokenProb {
        llama_token token;
        float prob;
        std::string text;
    };
    std::vector<std::vector<TokenProb>> all_token_probs; // For each generated token

    while (n_generated < max_tokens) {
        // Sample the next token
        new_token_id = llama_sampler_sample(data->sampler, data->ctx, -1);

        // Store token probabilities if requested
        if (n_probs > 0) {
            std::vector<TokenProb> top_probs;
            const float* logits = llama_get_logits_ith(data->ctx, -1);

            // Create a sorted list of token probabilities
            std::vector<std::pair<llama_token, float>> token_probs;
            for (llama_token token_id = 0; token_id < llama_vocab_n_tokens(vocab); token_id++) {
                token_probs.push_back({token_id, logits[token_id]});
            }

            // Sort by logit value (descending)
            std::sort(token_probs.begin(), token_probs.end(),
                     [](const auto& a, const auto& b) { return a.second > b.second; });

            // Apply softmax to get probabilities for top n_probs tokens
            float max_logit = token_probs[0].second;
            float sum_exp = 0.0f;

            // Calculate sum for softmax
            for (int i = 0; i < std::min(n_probs, (int)token_probs.size()); i++) {
                sum_exp += expf(token_probs[i].second - max_logit);
            }

            // Store top n_probs tokens with their probabilities
            for (int i = 0; i < std::min(n_probs, (int)token_probs.size()); i++) {
                TokenProb tp;
                tp.token = token_probs[i].first;
                tp.prob = expf(token_probs[i].second - max_logit) / sum_exp;

                // Get token text
                char buf[256];
                int n = llama_token_to_piece(vocab, tp.token, buf, sizeof(buf), 0, !no_parse_special_tokens);
                if (n > 0) {
                    tp.text = std::string(buf, n);
                }

                top_probs.push_back(tp);
            }

            all_token_probs.push_back(top_probs);

            if (verbose && n_generated < 5) { // Show first few tokens
                fprintf(stderr, "\nTop %d probabilities for token %d:\n", n_probs, n_generated);
                for (const auto& tp : top_probs) {
                    fprintf(stderr, "  Token %d ('%s'): %.4f\n",
                            tp.token, tp.text.c_str(), tp.prob);
                }
            }
        }

        // Is it an end of generation? Check against custom EOS if provided
        if (!ignore_eos) {
            if (custom_eos != default_eos && new_token_id == custom_eos) {
                break; // Custom EOS token
            } else if (llama_vocab_is_eog(vocab, new_token_id)) {
                break; // Default EOG tokens
            }
        }

        // Convert the token to a string
        char buf[256];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, !no_parse_special_tokens);
        if (n < 0) {
            set_last_error("Failed to convert token to piece");
            break;
        }
        std::string piece(buf, n);
        response += piece;
        n_generated++;

        // Add to all_tokens
        data->all_tokens.push_back(new_token_id);

        // Check stop sequences
        if (!stop_sequences.empty()) {
            bool should_stop = false;
            for (const auto& stop_seq : stop_sequences) {
                if (response.size() >= stop_seq.size()) {
                    if (response.substr(response.size() - stop_seq.size()) == stop_seq) {
                        should_stop = true;
                        // Remove stop sequence from response
                        response = response.substr(0, response.size() - stop_seq.size());
                        break;
                    }
                }
            }
            if (should_stop) {
                break;
            }
        }

        // Stream callback
        if (tokenCallback) {
            jstring tokenJStr = env->NewStringUTF(piece.c_str());
            jboolean shouldContinue = env->CallBooleanMethod(tokenCallback, testMethod, tokenJStr);
            env->DeleteLocalRef(tokenJStr);

            if (!shouldContinue) {
                break;
            }
        }

        // Prepare the next batch with the sampled token
        // Create batch with proper position
        llama_batch batch = llama_batch_init(1, 0, 1);
        batch.n_tokens = 1;
        batch.token[0] = new_token_id;
        batch.pos[0] = data->n_past;  // Critical: position is current KV cache size
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;  // seq_id already allocated by llama_batch_init
        batch.logits[0] = 1;

        if (data->n_past + batch.n_tokens > n_ctx) {
            if (no_context_shift) {
                // Stop generation instead of shifting context
                if (verbose) {
                    fprintf(stderr, "Context full, stopping generation (no_context_shift=true)\n");
                }
                set_last_error("Context size exceeded, generation stopped");
                free(batch.seq_id[0]);
                llama_batch_free(batch);
                break;
            } else {
                // Default behavior - could implement context shifting here in future
                fprintf(stderr, "context size exceeded\n");
                set_last_error("Context size exceeded");
                free(batch.seq_id[0]);
                llama_batch_free(batch);
                break;
            }
        }

        int ret = llama_decode(data->ctx, batch);
        llama_batch_free(batch);

        if (ret != 0) {
            set_last_error("Failed to decode, ret = " + std::to_string(ret));
            break;
        }

        data->n_past += batch.n_tokens;
    }

    // Add the response to the messages only in chat mode
    if (chat_mode && add_assistant) {
        data->chat_messages.push_back({"assistant", strdup(response.c_str())});
    } else {
        // In plain text mode or when add_assistant is false, DON'T include the response in last_prompt
        // This allows the next call with the same base context to be detected as continuation
        // data->last_prompt is already set to complete_prompt above
    }

    // Session save/load handling
    if (!cache_session.empty()) {
        // Save session state after generation
        if (verbose) {
            fprintf(stderr, "\n=== Saving session to: %s ===\n", cache_session.c_str());
        }

        // Get all tokens processed (prompt + generated)
        std::vector<llama_token> session_tokens = data->all_tokens;

        // Save the session state
        bool save_success = llama_state_save_file(data->ctx, cache_session.c_str(),
                                                 session_tokens.data(), session_tokens.size());

        if (save_success) {
            if (verbose) {
                fprintf(stderr, "Session saved successfully (%zu tokens)\n", session_tokens.size());
            }
        } else {
            fprintf(stderr, "WARNING: Failed to save session to %s\n", cache_session.c_str());
        }
    }

    if (verbose) {
        fprintf(stderr, "\n=== Generation complete ===\n");
        fprintf(stderr, "Generated %d tokens\n", n_generated);
        fprintf(stderr, "Total tokens in context: %d\n", data->n_past);
        fprintf(stderr, "=========================\n");
    }

    return 0;
}

JNIEXPORT jstring JNICALL Java_org_llm_wrapper_LlamaCpp_llama_1get_1error
  (JNIEnv *env, jclass) {
    std::lock_guard<std::mutex> lock(g_error_mutex);
    return env->NewStringUTF(last_error_message.c_str());
}

JNIEXPORT jlong JNICALL Java_org_llm_wrapper_LlamaCpp_llama_1get_1native_1memory_1usage
  (JNIEnv * /* env */, jclass, jlong handle) {
    LlamaContextData* data = reinterpret_cast<LlamaContextData*>(handle);
    if (!is_valid_context(data)) {
        return -1;
    }

    // Get model size in bytes
    size_t model_size = 0;
    if (data->model) {
        model_size = llama_model_size(data->model);
    }

    // Get context memory usage
    size_t ctx_size = 0;
    if (data->ctx) {
        ctx_size = llama_state_get_size(data->ctx);
    }

    // Get OS-reported memory usage
    size_t os_memory = get_process_memory_usage();

    // Log memory details for debugging
    fprintf(stderr, "\n=== Memory Usage ===\n");
    fprintf(stderr, "Model size: %.2f MB\n", model_size / (1024.0 * 1024.0));
    fprintf(stderr, "Context size: %.2f MB\n", ctx_size / (1024.0 * 1024.0));
    fprintf(stderr, "OS-reported memory: %.2f MB\n", os_memory / (1024.0 * 1024.0));
    fprintf(stderr, "===================\n");

    // Return the maximum of calculated and OS-reported memory
    size_t calculated_total = model_size + ctx_size;
    return static_cast<jlong>(std::max(calculated_total, os_memory));
}

JNIEXPORT jlong JNICALL Java_org_llm_wrapper_LlamaCpp_llama_1get_1os_1memory_1usage
  (JNIEnv * /* env */, jclass) {
    size_t os_memory = get_process_memory_usage();
    return static_cast<jlong>(os_memory);
}

JNIEXPORT void JNICALL Java_org_llm_wrapper_LlamaCpp_llama_1clear_1chat
  (JNIEnv *, jclass, jlong handle) {
    LlamaContextData* data = reinterpret_cast<LlamaContextData*>(handle);
    if (!is_valid_context(data)) {
        return;
    }

    // Free existing chat messages
    free_chat_messages(data->chat_messages);

    // Clear formatted chat buffer
    data->formatted_chat.clear();

    // Clear token tracking based on n_keep
    if (data->n_keep == 0) {
        // Clear everything
        data->all_tokens.clear();
        data->n_past = 0;
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
        llama_kv_self_clear(data->ctx);
#ifdef __clang__
#pragma clang diagnostic pop
#endif
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
    } else if (data->n_keep > 0 && data->n_past > data->n_keep) {
        // Keep only first n_keep tokens
        data->all_tokens.resize(data->n_keep);
        data->n_past = data->n_keep;
        // Clear KV cache beyond n_keep
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
        llama_kv_self_seq_rm(data->ctx, 0, data->n_keep, -1);
#ifdef __clang__
#pragma clang diagnostic pop
#endif
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
    }
    // If n_keep is -1, keep everything (don't clear)
}
