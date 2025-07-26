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

        if (data->sampler) {
            llama_sampler_free(data->sampler);
        }
        if (data->ctx) {
            llama_ctx_free(data->ctx);
        }
        if (data->model) {
            llama_model_free(data->model);
        }
        delete data;
    }
    g_active_contexts.clear();
    llama_backend_free();
}

// --- Context tracking for safe cleanup ---

static void register_context(LlamaContextData* data) {
    std::lock_guard<std::mutex> lock(g_context_registry_mutex);
    g_active_contexts.insert(data);
}

static void unregister_context(LlamaContextData* data) {
    std::lock_guard<std::mutex> lock(g_context_registry_mutex);
    g_active_contexts.erase(data);
}

static bool is_valid_context(LlamaContextData* data) {
    std::lock_guard<std::mutex> lock(g_context_registry_mutex);
    return g_active_contexts.find(data) != g_active_contexts.end();
}

// --- JNI utility functions ---

static jstring getStringField(JNIEnv *env, jobject obj, const char* fieldName) {
    jclass cls = env->GetObjectClass(obj);
    jfieldID fid = env->GetFieldID(cls, fieldName, "Ljava/lang/String;");
    return (jstring)env->GetObjectField(obj, fid);
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

static jobject getObjectField(JNIEnv *env, jobject obj, const char* fieldName, const char* sig) {
    jclass cls = env->GetObjectClass(obj);
    jfieldID fid = env->GetFieldID(cls, fieldName, sig);
    return env->GetObjectField(obj, fid);
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

    // Extract all parameters from InitParams
    data->params.n_ctx = getIntField(env, initParams, "nCtx");
    data->params.n_batch = getIntField(env, initParams, "nBatch");
    data->params.cpuparams.n_threads = getIntField(env, initParams, "nThreads");
    data->params.cpuparams_batch.n_threads = getIntField(env, initParams, "nThreadsBatch");

    // Note: GPU device selection through params.devices requires newer API
    // For now, we'll rely on n_gpu_layers for GPU usage

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

    // Get n_keep parameter for cache reuse
    data->n_keep = getIntField(env, initParams, "nKeep");

    // Set seed properly - only use LLAMA_DEFAULT_SEED if seed is explicitly -1
    data->params.sampling.seed = (seed == -1) ? LLAMA_DEFAULT_SEED : seed;

    // Set escape processing based on noEscape flag
    data->params.no_escape = no_escape;

    data->n_ctx = data->params.n_ctx;

    // Load model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    model_params.use_mmap = use_mmap;
    model_params.use_mlock = use_mlock;

    // Handle tensor splits if provided
    jfloatArray tensorSplits = getFloatArrayField(env, initParams, "tensorSplits");
    if (tensorSplits != nullptr) {
        jsize splitCount = env->GetArrayLength(tensorSplits);
        // Note: tensor_split is now const in newer versions
        // This feature may need to be implemented differently
        // For now, we'll skip it to avoid compilation errors
        // TODO: Implement tensor split support for newer API
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
    ctx_params.n_ubatch = data->params.n_batch;
    ctx_params.n_threads = data->params.cpuparams.n_threads;
    ctx_params.n_threads_batch = data->params.cpuparams_batch.n_threads;
    ctx_params.embeddings = embeddings_mode;
    ctx_params.flash_attn = true; // Enable flash attention

    // Set cache types if different from default
    if (cache_type_k != 1) { // 1 = GGML_TYPE_F16
        ctx_params.type_k = static_cast<enum ggml_type>(cache_type_k);
    }
    if (cache_type_v != 1) {
        ctx_params.type_v = static_cast<enum ggml_type>(cache_type_v);
    }

    data->ctx = llama_ctx_init_from_model(data->model, ctx_params);
    if (!data->ctx) {
        llama_model_free(data->model);
        std::cerr << "Failed to create context" << std::endl;
        set_last_error("Failed to create context from model");
        return 0;
    }

    // Register context for safe cleanup
    register_context(data.get());

    return reinterpret_cast<jlong>(data.release());
}

JNIEXPORT void JNICALL Java_org_llm_wrapper_LlamaCpp_llama_1destroy
  (JNIEnv *, jclass, jlong handle) {
    LlamaContextData* data = reinterpret_cast<LlamaContextData*>(handle);
    if (!is_valid_context(data)) {
        return;
    }

    unregister_context(data);

    // Free chat messages using helper function
    free_chat_messages(data->chat_messages);

    if (data->sampler) {
        llama_sampler_free(data->sampler);
    }
    if (data->ctx) {
        llama_ctx_free(data->ctx);
    }
    if (data->model) {
        llama_model_free(data->model);
    }
    delete data;
}

JNIEXPORT jint JNICALL Java_org_llm_wrapper_LlamaCpp_llama_1generate
  (JNIEnv *env, jclass, jlong handle, jobject generateParams) {
    LlamaContextData* data = reinterpret_cast<LlamaContextData*>(handle);
    if (!is_valid_context(data)) {
        set_last_error("Invalid context handle");
        return -1;
    }

    const char* prompt_cstr = nullptr;
    jstring prompt_jstr = getStringField(env, generateParams, "prompt");
    if (!prompt_jstr) {
        set_last_error("Prompt is null");
        return -1;
    }
    prompt_cstr = env->GetStringUTFChars(prompt_jstr, nullptr);

    // First message marker
    bool is_first = data->chat_messages.empty();
    std::string user_input(prompt_cstr);

    // Get generation parameters
    int n_predict = getIntField(env, generateParams, "nPredict");
    float temp = getFloatField(env, generateParams, "temp");
    int top_k = getIntField(env, generateParams, "topK");
    float top_p = getFloatField(env, generateParams, "topP");
    float repeat_penalty = getFloatField(env, generateParams, "repeatPenalty");
    int repeat_last_n = getIntField(env, generateParams, "repeatLastN");
    int seed = getIntField(env, generateParams, "seed");
    bool stream = getBooleanField(env, generateParams, "stream");

    // Apply default values for unset parameters
    if (top_k <= 0) top_k = 40;
    if (top_p <= 0.0f || top_p > 1.0f) top_p = 0.95f;
    if (temp < 0.0f) temp = 0.8f;
    if (repeat_penalty <= 0.0f) repeat_penalty = 1.1f;
    if (repeat_last_n <= 0) repeat_last_n = 64;
    if (n_predict <= 0) n_predict = 128;

    // Initialize vocab
    const llama_vocab * vocab = llama_model_get_vocab(data->model);

    // Create new sampler for each generation
    if (data->sampler) {
        llama_sampler_free(data->sampler);
        data->sampler = nullptr;
    }

    // Initialize sampler chain
    data->sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (!data->sampler) {
        env->ReleaseStringUTFChars(prompt_jstr, prompt_cstr);
        set_last_error("Failed to create sampler chain");
        return -1;
    }

    // Add samplers matching simple-chat.cpp
    {
        if (temp > 0.0f) {
            llama_sampler_chain_add(data->sampler, llama_sampler_init_greedy());
        } else {
            // Temperature sampling
            if (top_k > 1) {
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
        // Determine how many tokens we should keep
        int tokens_to_keep = data->n_keep;
        if (data->n_keep == -1) {
            // Keep all tokens
            tokens_to_keep = data->cached_token_count;
        } else {
            // Keep at most n_keep tokens
            tokens_to_keep = std::min(data->n_keep, data->cached_token_count);
        }

        // Compare tokens to find common prefix, but only up to tokens_to_keep
        int max_compare = std::min(tokens_to_keep, n_prompt_tokens);
        for (int i = 0; i < max_compare; i++) {
            if (data->cached_tokens[i] == prompt_tokens[i]) {
                common_prefix_len++;
            } else {
                break;
            }
        }

        fprintf(stderr, "\n=== Cache reuse analysis ===\n");
        fprintf(stderr, "n_keep: %d\n", data->n_keep);
        fprintf(stderr, "Cached tokens: %d\n", data->cached_token_count);
        fprintf(stderr, "Tokens to keep: %d\n", tokens_to_keep);
        fprintf(stderr, "New prompt tokens: %d\n", n_prompt_tokens);
        fprintf(stderr, "Common prefix length: %d\n", common_prefix_len);
        fprintf(stderr, "Tokens to process: %d\n", n_prompt_tokens - common_prefix_len);
        fprintf(stderr, "===========================\n");

        // If we have a common prefix, we can skip processing those tokens
        if (common_prefix_len > 0) {
            // Remove non-matching tokens from KV cache
            int tokens_to_remove = data->cached_token_count - common_prefix_len;
            if (tokens_to_remove > 0) {
                // Clear KV cache from the divergence point
                llama_kv_self_seq_rm(data->ctx, 0, common_prefix_len, -1);
            }
        } else if (data->n_keep == 0 || data->cached_token_count > tokens_to_keep) {
            // If n_keep is 0 or we have more cached tokens than we should keep, clear cache
            llama_kv_self_clear(data->ctx);
            data->cached_token_count = 0;
        }
    } else if (is_first && data->n_keep == 0) {
        // If n_keep is 0 and this is the first message, ensure cache is clear
        llama_kv_self_clear(data->ctx);
        data->cached_token_count = 0;
    }

    // Update cached tokens based on n_keep setting
    if (data->n_keep != 0) {
        data->cached_tokens = prompt_tokens;
        data->cached_token_count = n_prompt_tokens;

        // Trim cached tokens if n_keep has a specific limit
        if (data->n_keep > 0 && data->cached_token_count > data->n_keep) {
            data->cached_tokens.resize(data->n_keep);
            data->cached_token_count = data->n_keep;
        }
    } else {
        // n_keep is 0, don't cache tokens
        data->cached_tokens.clear();
        data->cached_token_count = 0;
    }

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

        // Clear the batch
        llama_batch batch = llama_batch_init(batch_size, 0, 1);

        // Add tokens to the batch
        for (int j = 0; j < batch_size; j++) {
            llama_batch_add(batch, prompt_tokens[i + j], i + j, { 0 }, false);
        }

        // Mark the last token as needing logits
        batch.logits[batch.n_tokens - 1] = true;

        fprintf(stderr, "Processing batch: tokens %d-%d\n", i, i + batch_size - 1);

        // Decode the batch
        if (llama_decode(data->ctx, batch) != 0) {
            llama_batch_free(batch);
            set_last_error("Failed to decode prompt batch");
            return -1;
        }

        llama_batch_free(batch);
    }

    // Generate response tokens
    int n_generated = 0;
    const int max_context_size = n_ctx;
    int n_consumed = n_prompt_tokens;

    fprintf(stderr, "\n=== Generation phase ===\n");
    fprintf(stderr, "Max tokens to generate: %d\n", n_predict);
    fprintf(stderr, "Context consumed: %d/%d\n", n_consumed, max_context_size);

    while (n_generated < n_predict) {
        // Sample next token
        const llama_token new_token_id = llama_sampler_sample(data->sampler, data->ctx, -1);

        // Accept and add to context
        llama_sampler_accept(data->sampler, new_token_id);

        // Check for end of generation
        if (llama_token_is_eog(vocab, new_token_id)) {
            fprintf(stderr, "End of generation token detected\n");
            break;
        }

        // Convert token to string
        std::string piece = llama_token_to_piece(data->ctx, new_token_id, true);
        response += piece;

        // Call Java callback if provided
        if (tokenCallback) {
            jstring token_jstr = env->NewStringUTF(piece.c_str());
            jboolean should_continue = env->CallBooleanMethod(tokenCallback, testMethod, token_jstr);
            env->DeleteLocalRef(token_jstr);

            if (!should_continue) {
                fprintf(stderr, "Generation stopped by callback\n");
                break;
            }
        }

        // Print token if streaming
        if (stream) {
            printf("%s", piece.c_str());
            fflush(stdout);
        }

        // Check context size
        if (++n_consumed >= max_context_size) {
            fprintf(stderr, "Context size limit reached\n");
            break;
        }

        // Prepare batch for next iteration
        llama_batch batch = llama_batch_init(1, 0, 1);
        llama_batch_add(batch, new_token_id, n_consumed - 1, { 0 }, true);

        if (llama_decode(data->ctx, batch) != 0) {
            llama_batch_free(batch);
            set_last_error("Failed to decode during generation");
            return -1;
        }

        llama_batch_free(batch);
        n_generated++;
    }

    // Add assistant response to chat history
    data->chat_messages.push_back({"assistant", strdup(response.c_str())});

    // Update prev_len for next iteration
    data->prev_len = new_len;

    fprintf(stderr, "\n=== Generation complete ===\n");
    fprintf(stderr, "Generated tokens: %d\n", n_generated);
    fprintf(stderr, "Response length: %zu chars\n", response.length());
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
        llama_kv_self_clear(data->ctx);
    } else if (data->n_keep > 0 && data->cached_token_count > data->n_keep) {
        // Keep only first n_keep tokens
        data->cached_tokens.resize(data->n_keep);
        data->cached_token_count = data->n_keep;
        // Clear KV cache beyond n_keep
        llama_kv_self_seq_rm(data->ctx, 0, data->n_keep, -1);
    }
    // If n_keep is -1, keep all cached tokens (don't clear)
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#ifdef __cplusplus
}
#endif
