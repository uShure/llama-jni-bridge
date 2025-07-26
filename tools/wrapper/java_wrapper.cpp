#include "org_llm_wrapper_LlamaCpp.h"
#include "llama.h"
#include "common.h"
#include <iostream>
#include <string>
#include <vector>
#include <unordered_set>
#include <mutex>
#include <memory>
#include <thread>
#include <cstring>

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
    int prev_len = 0;  // Track previous formatted length

    // Token tracking for cache reuse
    std::vector<llama_token> cached_tokens;  // Tokens currently in KV cache
    int cached_token_count = 0;  // Number of valid tokens in cache
    int n_keep = 0;  // Number of tokens to keep from initial prompt (-1 = all)
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
    jfieldID fid = env->GetFieldID(cls, fieldName, "I");
    return env->GetIntField(obj, fid);
}

static jfloat getFloatField(JNIEnv *env, jobject obj, const char* fieldName) {
    jclass cls = env->GetObjectClass(obj);
    jfieldID fid = env->GetFieldID(cls, fieldName, "F");
    return env->GetFloatField(obj, fid);
}

static jboolean getBooleanField(JNIEnv *env, jobject obj, const char* fieldName) {
    jclass cls = env->GetObjectClass(obj);
    jfieldID fid = env->GetFieldID(cls, fieldName, "Z");
    return env->GetBooleanField(obj, fid);
}

static jstring getStringField(JNIEnv *env, jobject obj, const char* fieldName) {
    jclass cls = env->GetObjectClass(obj);
    jfieldID fid = env->GetFieldID(cls, fieldName, "Ljava/lang/String;");
    return (jstring)env->GetObjectField(obj, fid);
}

static jobject getObjectField(JNIEnv *env, jobject obj, const char* fieldName, const char* sig) {
    jclass cls = env->GetObjectClass(obj);
    jfieldID fid = env->GetFieldID(cls, fieldName, sig);
    return env->GetObjectField(obj, fid);
}

static jintArray getIntArrayField(JNIEnv *env, jobject obj, const char* fieldName) {
    jclass cls = env->GetObjectClass(obj);
    jfieldID fid = env->GetFieldID(cls, fieldName, "[I");
    return (jintArray)env->GetObjectField(obj, fid);
}

static jfloatArray getFloatArrayField(JNIEnv *env, jobject obj, const char* fieldName) {
    jclass cls = env->GetObjectClass(obj);
    jfieldID fid = env->GetFieldID(cls, fieldName, "[F");
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

    // Extract parameters
    data->params.n_ctx = getIntField(env, initParams, "nCtx");
    data->params.n_batch = getIntField(env, initParams, "nBatch");
    data->params.cpuparams.n_threads = getIntField(env, initParams, "nThreads");
    data->params.cpuparams_batch.n_threads = getIntField(env, initParams, "nThreadsBatch");

    // Handle GPU devices properly
    // Note: The devices parameter might not be directly supported in newer API versions
    // Instead, GPU usage is typically controlled via n_gpu_layers and tensor_split

    jintArray gpuDevices = getIntArrayField(env, initParams, "gpuDevices");
    if (gpuDevices != nullptr) {
        jsize deviceCount = env->GetArrayLength(gpuDevices);
        if (deviceCount > 0) {
            jint* devices = env->GetIntArrayElements(gpuDevices, nullptr);

            // Log the requested devices
            fprintf(stderr, "Requested GPU devices: ");
            for (jsize i = 0; i < deviceCount; i++) {
                fprintf(stderr, "%d ", devices[i]);
            }
            fprintf(stderr, "\n");

            // Note: In newer llama.cpp, device selection is often automatic
            // based on n_gpu_layers. The specific device IDs might not be
            // directly configurable through the context parameters.
            // This is logged for debugging purposes.

            env->ReleaseIntArrayElements(gpuDevices, devices, JNI_ABORT);
        }
    }
    // If gpuDevices is null, use default GPU selection based on n_gpu_layers

    // Match simple-chat.cpp: if n_batch is 0, set it to n_ctx
    if (data->params.n_batch == 0) {
        data->params.n_batch = data->params.n_ctx;
    }

    int n_gpu_layers = getIntField(env, initParams, "nGpuLayers");
    int seed = getIntField(env, initParams, "seed");
    bool use_mmap = getBooleanField(env, initParams, "useMmap");
    bool use_mlock = getBooleanField(env, initParams, "useMlock");
    bool embeddings_mode = getBooleanField(env, initParams, "embeddings");

    // Get cache type parameters
    int cache_type_k = getIntField(env, initParams, "cacheTypeK");
    int cache_type_v = getIntField(env, initParams, "cacheTypeV");
    bool no_escape = getBooleanField(env, initParams, "noEscape");

    // Add support for draft model cache types
    int cache_type_k_draft = getIntField(env, initParams, "cacheTypeKDraft");
    int cache_type_v_draft = getIntField(env, initParams, "cacheTypeVDraft");

    // Note: Draft cache types would be used when loading draft models for speculative decoding
    // Currently stored for future use when draft model support is added
    (void)cache_type_k_draft; // Suppress unused warning
    (void)cache_type_v_draft; // Suppress unused warning

    // Get n_keep parameter for cache reuse
    data->n_keep = getIntField(env, initParams, "nKeep");

    // Set escape processing based on noEscape flag
    data->params.escape = !no_escape;  // escape is opposite of noEscape

    // Set seed properly - only use LLAMA_DEFAULT_SEED if seed is explicitly -1
    data->params.sampling.seed = (seed == -1) ? LLAMA_DEFAULT_SEED : seed;

    data->n_ctx = data->params.n_ctx;

    // Load model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    model_params.use_mmap = use_mmap;
    model_params.use_mlock = use_mlock;

    // Handle tensor splits if provided
    jfloatArray tensorSplits = getFloatArrayField(env, initParams, "tensorSplits");
    if (tensorSplits != nullptr) {
        // jsize splitCount = env->GetArrayLength(tensorSplits);
        // Note: tensor_split is const in newer API, skip for now
        // TODO: Find alternative way to set tensor splits
        /*
        if (splitCount > 0 && splitCount <= LLAMA_MAX_DEVICES) {
            jfloat* splits = env->GetFloatArrayElements(tensorSplits, nullptr);
            for (jsize i = 0; i < splitCount && i < LLAMA_MAX_DEVICES; i++) {
                model_params.tensor_split[i] = splits[i];
            }
            env->ReleaseFloatArrayElements(tensorSplits, splits, JNI_ABORT);
        }
        */
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
    ctx_params.n_threads = data->params.cpuparams.n_threads;
    ctx_params.n_threads_batch = data->params.cpuparams_batch.n_threads;
    ctx_params.embeddings = embeddings_mode;
    ctx_params.type_k = (ggml_type)cache_type_k;
    ctx_params.type_v = (ggml_type)cache_type_v;

    data->ctx = llama_init_from_model(data->model, ctx_params);
    if (!data->ctx) {
        std::cerr << "Failed to create context" << std::endl;
        set_last_error("Failed to create context from model");
        llama_model_free(data->model);
        return 0;
    }

    // Initialize sampler with simple-chat.cpp parameters, using proper seed handling
    data->sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(data->sampler, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(data->sampler, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(data->sampler, llama_sampler_init_dist(data->params.sampling.seed));

    // Initialize formatted_chat buffer with context size
    data->formatted_chat.resize(data->n_ctx);

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

    int n_predict = getIntField(env, generateParams, "nPredict");
    int top_k = getIntField(env, generateParams, "topK");
    float top_p = getFloatField(env, generateParams, "topP");
    float temp = getFloatField(env, generateParams, "temp");
    float repeat_penalty = getFloatField(env, generateParams, "repeatPenalty");
    int repeat_last_n = getIntField(env, generateParams, "repeatLastN");
    bool stream = getBooleanField(env, generateParams, "stream");
    int seed = getIntField(env, generateParams, "seed");

    // Get vocab from model
    const struct llama_vocab * vocab = llama_model_get_vocab(data->model);

    // Check if this is first prompt in the conversation
    const bool is_first = llama_memory_seq_pos_max(llama_get_memory(data->ctx), 0) == -1;

    // Get keep parameter from GenerateParams to allow per-generation control
    int keep = getIntField(env, generateParams, "keep");

    // If keep is specified in GenerateParams and different from the init value, update it
    if (keep != -1 && keep != data->n_keep) {
        fprintf(stderr, "Overriding n_keep: %d -> %d for this generation\n", data->n_keep, keep);
        // Override n_keep for this generation
        data->n_keep = keep;

        // If we're disabling caching (keep=0) and have cached tokens, clear them
        if (keep == 0 && data->cached_token_count > 0) {
            llama_memory_t mem = llama_get_memory(data->ctx);
            llama_memory_clear(mem, false);
            data->cached_tokens.clear();
            data->cached_token_count = 0;
        }
    }

    // Debug output
    fprintf(stderr, "\n=== Generate parameters ===\n");
    fprintf(stderr, "n_predict: %d\n", n_predict);
    fprintf(stderr, "temp: %.2f\n", temp);
    fprintf(stderr, "top_k: %d\n", top_k);
    fprintf(stderr, "top_p: %.2f\n", top_p);
    fprintf(stderr, "repeat_penalty: %.2f\n", repeat_penalty);
    fprintf(stderr, "repeat_last_n: %d\n", repeat_last_n);
    fprintf(stderr, "seed: %d\n", seed);
    fprintf(stderr, "stream: %s\n", stream ? "true" : "false");
    fprintf(stderr, "=== End parameters ===\n\n");

    // Reset sampler for new generation
    llama_sampler_reset(data->sampler);

    // Check if we need to recreate sampler with different parameters
    // Default values: temp=0.8, top_k=40, top_p=0.9, repeat_penalty=1.1
    bool use_default_sampler = (temp == 0.8f && top_k == 40 && top_p == 0.9f &&
                                repeat_penalty == 1.1f && seed == -1);

    if (!use_default_sampler) {
        // Recreate sampler chain with user-provided parameters
        llama_sampler_free(data->sampler);
        data->sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());

        // Add samplers to match llama-cli behavior
        if (top_k > 0 && top_k < llama_vocab_n_tokens(vocab)) {
            llama_sampler_chain_add(data->sampler, llama_sampler_init_top_k(top_k));
        }
        if (top_p < 1.0f && top_p > 0.0f) {
            llama_sampler_chain_add(data->sampler, llama_sampler_init_top_p(top_p, 1));
        }
        llama_sampler_chain_add(data->sampler, llama_sampler_init_min_p(0.05f, 1));
        llama_sampler_chain_add(data->sampler, llama_sampler_init_temp(temp));
        if (repeat_penalty != 1.0f && repeat_last_n > 0) {
            llama_sampler_chain_add(data->sampler, llama_sampler_init_penalties(
                repeat_last_n,    // penalty_last_n
                repeat_penalty,   // penalty_repeat
                0.0f,            // penalty_freq (disabled)
                0.0f             // penalty_present (disabled)
            ));
        }
        // Use the seed from parameters, only use LLAMA_DEFAULT_SEED if seed is -1
        uint32_t actual_seed = (seed == -1) ? LLAMA_DEFAULT_SEED : (uint32_t)seed;
        llama_sampler_chain_add(data->sampler, llama_sampler_init_dist(actual_seed));
    }

    // Get chat template
    const char * tmpl = llama_model_chat_template(data->model, nullptr);

    // Add user message to chat history
    data->chat_messages.push_back({"user", strdup(user_input.c_str())});

    // Format messages with chat template
    if (data->formatted_chat.empty()) {
        data->formatted_chat.resize(data->n_ctx);
    }

    int new_len;
    if (tmpl == nullptr) {
        // No template found - use simple format for Llama-2
        fprintf(stderr, "No chat template found, using simple format\n");

        // For first message, just use the prompt directly
        if (is_first) {
            new_len = snprintf(data->formatted_chat.data(), data->formatted_chat.size(),
                             "%s", user_input.c_str());
        } else {
            // For subsequent messages, append to existing conversation
            std::string formatted_messages;
            for (size_t i = 0; i < data->chat_messages.size() - 1; i++) {
                formatted_messages += data->chat_messages[i].content;
                formatted_messages += "\n";
            }
            formatted_messages += user_input;

            new_len = snprintf(data->formatted_chat.data(), data->formatted_chat.size(),
                             "%s", formatted_messages.c_str());
        }
    } else {
        // Use the model's chat template
        new_len = llama_chat_apply_template(tmpl, data->chat_messages.data(),
                                           data->chat_messages.size(), true,
                                           data->formatted_chat.data(), data->formatted_chat.size());

        if (new_len > (int)data->formatted_chat.size()) {
            data->formatted_chat.resize(new_len);
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

    // Extract only the new part of the prompt (remove previous content)
    std::string prompt(data->formatted_chat.begin() + data->prev_len,
                      data->formatted_chat.begin() + new_len);

    // Log the formatted prompt for debugging
    fprintf(stderr, "\n=== Debug: Chat template application ===\n");
    fprintf(stderr, "Template: %s\n", tmpl ? tmpl : "null");
    fprintf(stderr, "Messages count: %zu\n", data->chat_messages.size());
    fprintf(stderr, "Previous length: %d\n", data->prev_len);
    fprintf(stderr, "New length: %d\n", new_len);
    fprintf(stderr, "Is first message: %s\n", is_first ? "true" : "false");
    fprintf(stderr, "\n=== Formatted prompt (new part only) ===\n");
    fprintf(stderr, "%s", prompt.c_str());
    fprintf(stderr, "\n=== End of formatted prompt (length: %zu) ===\n", prompt.length());

    // Log first few tokens for debugging
    if (prompt.length() > 0) {
        fprintf(stderr, "First 10 chars (hex): ");
        for (size_t i = 0; i < std::min(prompt.length(), size_t(10)); i++) {
            fprintf(stderr, "%02x ", (unsigned char)prompt[i]);
        }
        fprintf(stderr, "\n");
    }

    env->ReleaseStringUTFChars(prompt_jstr, prompt_cstr);

    // Tokenize the new prompt part
    const int n_prompt_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                                              NULL, 0, is_first, true);
    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                      prompt_tokens.data(), prompt_tokens.size(), is_first, true) < 0) {
        set_last_error("Failed to tokenize the prompt");
        return -1;
    }

    // Get token callback
    jobject tokenCallback = getObjectField(env, generateParams, "tokenCallback", "Ljava/util/function/Predicate;");
    jclass predicateClass = nullptr;
    jmethodID testMethod = nullptr;

    if (tokenCallback) {
        predicateClass = env->GetObjectClass(tokenCallback);
        testMethod = env->GetMethodID(predicateClass, "test", "(Ljava/lang/Object;)Z");
    }

    // Cache reuse logic: find common prefix with cached tokens
    int common_prefix_len = 0;

    // Only reuse cache if n_keep is enabled (not 0)
    if (!is_first && data->cached_token_count > 0 && data->n_keep != 0) {
        // For large prompts with small questions, we want to find the common prefix
        // between the old and new token sequences

        // Compare tokens to find the longest common prefix
        int max_compare = std::min(data->cached_token_count, n_prompt_tokens);
        for (int i = 0; i < max_compare; i++) {
            if (data->cached_tokens[i] == prompt_tokens[i]) {
                common_prefix_len++;
            } else {
                break;  // Stop at first mismatch
            }
        }

        // If n_keep is set to a specific value, limit the prefix length
        if (data->n_keep > 0) {
            common_prefix_len = std::min(common_prefix_len, data->n_keep);
        }
        // If n_keep is -1, use all common prefix tokens

        fprintf(stderr, "\n=== Cache reuse analysis ===\n");
        fprintf(stderr, "n_keep: %d\n", data->n_keep);
        fprintf(stderr, "Cached tokens: %d\n", data->cached_token_count);
        fprintf(stderr, "New prompt tokens: %d\n", n_prompt_tokens);
        fprintf(stderr, "Common prefix length: %d\n", common_prefix_len);
        fprintf(stderr, "Tokens to process: %d\n", n_prompt_tokens - common_prefix_len);
        fprintf(stderr, "===========================\n");

        // If we have a common prefix, we can skip processing those tokens
        if (common_prefix_len > 0 && common_prefix_len < data->cached_token_count) {
            // Remove non-matching tokens from KV cache
            llama_memory_t mem = llama_get_memory(data->ctx);
            llama_memory_seq_rm(mem, 0, common_prefix_len, -1);
        } else if (common_prefix_len == 0) {
            // No common prefix, clear the cache
            llama_memory_t mem = llama_get_memory(data->ctx);
            llama_memory_clear(mem, false);
        }
        // If common_prefix_len == cached_token_count, all cached tokens match, keep them all
    } else if (is_first && data->n_keep == 0) {
        // If n_keep is 0 and this is the first message, ensure cache is clear
        llama_memory_t mem = llama_get_memory(data->ctx);
        llama_memory_clear(mem, false);
        data->cached_token_count = 0;
    }

    // Update cached tokens to the new prompt tokens
    data->cached_tokens = prompt_tokens;
    data->cached_token_count = n_prompt_tokens;

    // Process prompt tokens in batches (matching simple-chat.cpp)
    int n_batch = llama_n_batch(data->ctx);
    int n_ctx = llama_n_ctx(data->ctx);
    std::string response;

    fprintf(stderr, "\n=== Batch processing ===\n");
    fprintf(stderr, "Prompt tokens: %d\n", n_prompt_tokens);
    fprintf(stderr, "Batch size: %d\n", n_batch);
    fprintf(stderr, "Context size: %d\n", n_ctx);
    fprintf(stderr, "Starting from token: %d\n", common_prefix_len);

    // First, process the prompt tokens in batches, skipping common prefix
    for (int i = common_prefix_len; i < n_prompt_tokens; i += n_batch) {
        int batch_size = std::min(n_batch, n_prompt_tokens - i);
        llama_batch batch = llama_batch_get_one(prompt_tokens.data() + i, batch_size);

        // Check if we have enough space in the context
        int n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(data->ctx), 0) + 1;
        fprintf(stderr, "Processing batch %d-%d, context used: %d/%d\n",
                i, i + batch_size, n_ctx_used, n_ctx);

        if (n_ctx_used + batch.n_tokens > n_ctx) {
            fprintf(stderr, "context size exceeded: used %d + batch %d > max %d\n",
                    n_ctx_used, batch.n_tokens, n_ctx);
            set_last_error("Context size exceeded");
            return -1;
        }

        int ret = llama_decode(data->ctx, batch);
        if (ret != 0) {
            set_last_error("Failed to decode prompt batch, ret = " + std::to_string(ret));
            return -1;
        }
    }
    fprintf(stderr, "=== End batch processing ===\n");

    // Now generate the response token by token
    llama_token new_token_id;
    int n_generated = 0;

    while (n_generated < n_predict) {
        // Sample the next token
        new_token_id = llama_sampler_sample(data->sampler, data->ctx, -1);

        // Is it an end of generation?
        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        // Convert the token to a string
        char buf[256];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            set_last_error("Failed to convert token to piece");
            break;
        }
        std::string piece(buf, n);
        response += piece;
        n_generated++;

        // Stream callback
        if (stream && tokenCallback) {
            jstring tokenJStr = env->NewStringUTF(piece.c_str());
            jboolean shouldContinue = env->CallBooleanMethod(tokenCallback, testMethod, tokenJStr);
            env->DeleteLocalRef(tokenJStr);

            if (!shouldContinue) {
                break;
            }
        }

        // Prepare the next batch with the sampled token
        llama_batch batch = llama_batch_get_one(&new_token_id, 1);

        // Check context size again
        int n_ctx = llama_n_ctx(data->ctx);
        int n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(data->ctx), 0) + 1;
        if (n_ctx_used + batch.n_tokens > n_ctx) {
            fprintf(stderr, "context size exceeded\n");
            set_last_error("Context size exceeded");
            break;
        }

        int ret = llama_decode(data->ctx, batch);
        if (ret != 0) {
            set_last_error("Failed to decode, ret = " + std::to_string(ret));
            break;
        }
    }

    // Add the response to the messages
    data->chat_messages.push_back({"assistant", strdup(response.c_str())});

    // Update prev_len to point to end of all messages (without adding assistant prefix)
    // This matches simple-chat.cpp behavior
    if (tmpl == nullptr) {
        // For simple format, calculate the length of all messages
        data->prev_len = 0;
        for (const auto& msg : data->chat_messages) {
            data->prev_len += strlen(msg.content) + 1; // +1 for newline
        }
    } else {
        data->prev_len = llama_chat_apply_template(tmpl, data->chat_messages.data(),
                                                  data->chat_messages.size(), false,
                                                  nullptr, 0);
        if (data->prev_len < 0) {
            set_last_error("Failed to apply the chat template");
            return -1;
        }
    }

    fprintf(stderr, "\n=== Generation complete ===\n");
    fprintf(stderr, "Response length: %zu tokens\n", response.length());
    fprintf(stderr, "Updated prev_len: %d\n", data->prev_len);
    fprintf(stderr, "Total messages: %zu\n", data->chat_messages.size());
    fprintf(stderr, "=========================\n");

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

    // Free existing chat messages using helper function
    free_chat_messages(data->chat_messages);

    // Clear formatted chat buffer
    data->formatted_chat.clear();
    data->prev_len = 0;

    // Clear cached tokens based on n_keep setting
    if (data->n_keep == 0) {
        // n_keep is 0, clear everything
        data->cached_tokens.clear();
        data->cached_token_count = 0;
        llama_memory_t mem = llama_get_memory(data->ctx);
        llama_memory_clear(mem, false);
    } else if (data->n_keep > 0 && data->cached_token_count > data->n_keep) {
        // Keep only first n_keep tokens
        data->cached_tokens.resize(data->n_keep);
        data->cached_token_count = data->n_keep;
        // Clear KV cache beyond n_keep
        llama_memory_t mem = llama_get_memory(data->ctx);
        llama_memory_seq_rm(mem, 0, data->n_keep, -1);
    }
    // If n_keep is -1, keep all cached tokens (don't clear)
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif
