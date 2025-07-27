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

    // Token tracking for proper cache reuse
    std::vector<llama_token> all_tokens;  // All tokens ever processed
    int n_past = 0;  // Number of tokens in KV cache
    int n_keep = 0;  // Number of tokens to keep from initial prompt
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

    // Extract ALL parameters from InitParams
    data->params.n_ctx = getIntField(env, initParams, "nCtx");
    data->params.n_batch = getIntField(env, initParams, "nBatch");
    data->params.n_ubatch = getIntField(env, initParams, "nUBatch");
    data->params.cpuparams.n_threads = getIntField(env, initParams, "nThreads");
    data->params.cpuparams_batch.n_threads = getIntField(env, initParams, "nThreadsBatch");

    // Match simple-chat.cpp: if n_batch is 0, set it to n_ctx
    if (data->params.n_batch == 0) {
        data->params.n_batch = data->params.n_ctx;
    }

    int n_gpu_layers = getIntField(env, initParams, "nGpuLayers");
    int seed = getIntField(env, initParams, "seed");
    bool use_mmap = getBooleanField(env, initParams, "useMmap");
    bool use_mlock = getBooleanField(env, initParams, "useMlock");
    bool embeddings_mode = getBooleanField(env, initParams, "embeddings");
    bool flash_attn = getBooleanField(env, initParams, "flashAttn");
    bool no_kv_offload = getBooleanField(env, initParams, "noKVOffload");

    // Get cache type parameters
    int cache_type_k = getIntField(env, initParams, "cacheTypeK");
    int cache_type_v = getIntField(env, initParams, "cacheTypeV");
    bool no_escape = getBooleanField(env, initParams, "noEscape");

    // Get n_keep parameter for cache reuse
    data->n_keep = getIntField(env, initParams, "nKeep");

    // Set escape processing based on noEscape flag
    data->params.escape = !no_escape;

    // Set seed properly - only use LLAMA_DEFAULT_SEED if seed is explicitly -1
    data->params.sampling.seed = (seed == -1) ? LLAMA_DEFAULT_SEED : seed;

    data->n_ctx = data->params.n_ctx;

    // Load model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    model_params.use_mmap = use_mmap;
    model_params.use_mlock = use_mlock;

    // Get additional model parameters
    int main_gpu = getIntField(env, initParams, "mainGpu");
    model_params.main_gpu = main_gpu;

    // Handle tensor splits if provided
    jfloatArray tensorSplits = getFloatArrayField(env, initParams, "tensorSplits");
    if (tensorSplits != nullptr) {
        jsize splitCount = env->GetArrayLength(tensorSplits);
        if (splitCount > 0) {
            jfloat* splits = env->GetFloatArrayElements(tensorSplits, nullptr);
            // Note: Can't modify const tensor_split in newer API
            // This would need to be handled differently in newer llama.cpp versions
            env->ReleaseFloatArrayElements(tensorSplits, splits, JNI_ABORT);
        }
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
    ctx_params.no_perf = false;
    ctx_params.type_k = (ggml_type)cache_type_k;
    ctx_params.type_v = (ggml_type)cache_type_v;

    // Get additional context parameters
    float defrag_threshold = getFloatField(env, initParams, "defragThreshold");
    if (defrag_threshold > 0) {
        ctx_params.defrag_thold = defrag_threshold;
    }

    // No KV offload parameter
    ctx_params.offload_kqv = !no_kv_offload;

    data->ctx = llama_init_from_model(data->model, ctx_params);
    if (!data->ctx) {
        std::cerr << "Failed to create context" << std::endl;
        set_last_error("Failed to create context from model");
        llama_model_free(data->model);
        return 0;
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
    int n_predict = getIntField(env, generateParams, "nPredict");

    // Basic sampling parameters
    float temp = getFloatField(env, generateParams, "temp");
    int top_k = getIntField(env, generateParams, "topK");
    float top_p = getFloatField(env, generateParams, "topP");
    float min_p = getFloatField(env, generateParams, "minP");
    float tfs_z = getFloatField(env, generateParams, "tfsZ");
    float typical_p = getFloatField(env, generateParams, "typicalP");
    int seed = getIntField(env, generateParams, "seed");

    // Repetition penalty parameters
    float repeat_penalty = getFloatField(env, generateParams, "repeatPenalty");
    int repeat_last_n = getIntField(env, generateParams, "repeatLastN");
    float frequency_penalty = getFloatField(env, generateParams, "frequencyPenalty");
    float presence_penalty = getFloatField(env, generateParams, "presencePenalty");
    bool penalize_nl = getBooleanField(env, generateParams, "penalizeNl");

    // Mirostat parameters
    int mirostat = getIntField(env, generateParams, "mirostat");
    float mirostat_tau = getFloatField(env, generateParams, "mirostatTau");
    float mirostat_eta = getFloatField(env, generateParams, "mirostatEta");

    // DRY parameters
    float dry_multiplier = getFloatField(env, generateParams, "dryMultiplier");
    float dry_base = getFloatField(env, generateParams, "dryBase");
    int dry_allowed_length = getIntField(env, generateParams, "dryAllowedLength");
    int dry_penalty_last_n = getIntField(env, generateParams, "dryPenaltyLastN");

    // XTC parameters
    float xtc_threshold = getFloatField(env, generateParams, "xtcThreshold");
    float xtc_probability = getFloatField(env, generateParams, "xtcProbability");

    // Dynamic temperature
    float dyn_temp_range = getFloatField(env, generateParams, "dynTempRange");
    float dyn_temp_exponent = getFloatField(env, generateParams, "dynTempExponent");

    // Other parameters
    int keep = getIntField(env, generateParams, "keep");
    bool ignore_eos = getBooleanField(env, generateParams, "ignoreEos");

    // Get vocab from model
    const struct llama_vocab * vocab = llama_model_get_vocab(data->model);

    // Get keep parameter from GenerateParams
    if (keep != -1 && keep != data->n_keep) {
        data->n_keep = keep;
    }

    // Debug output
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

    // Free old sampler and create new one with current parameters
    if (data->sampler) {
        llama_sampler_free(data->sampler);
    }

    // Create sampler chain with ALL parameters
    data->sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());

    // Add DRY sampler if enabled
    if (dry_multiplier > 0.0f) {
        const char* dry_breakers[] = {"\n", "###", "\\n"};
        size_t num_breakers = 3;

        // Get n_ctx_train from model
        int32_t n_ctx_train = llama_model_n_ctx_train(data->model);

        llama_sampler_chain_add(data->sampler,
            llama_sampler_init_dry(vocab, n_ctx_train, dry_multiplier, dry_base,
                                 dry_allowed_length, dry_penalty_last_n,
                                 dry_breakers, num_breakers));
    }

    // Add samplers in the correct order (matching llama-cli)
    llama_sampler_chain_add(data->sampler,
        llama_sampler_init_logit_bias(llama_vocab_n_tokens(vocab), 0, nullptr, nullptr));

    // Add penalties sampler if any penalty is set
    if (repeat_penalty != 1.0f || frequency_penalty != 0.0f || presence_penalty != 0.0f) {
        llama_sampler_chain_add(data->sampler,
            llama_sampler_init_penalties(
                repeat_last_n,
                repeat_penalty,
                frequency_penalty,
                presence_penalty
            ));
    }

    // Top-K sampling
    if (top_k > 0) {
        llama_sampler_chain_add(data->sampler, llama_sampler_init_top_k(top_k));
    }

    // Tail-free sampling
    if (tfs_z < 1.0f) {
        llama_sampler_chain_add(data->sampler, llama_sampler_init_tail_free(tfs_z, 1));
    }

    // Typical sampling
    if (typical_p < 1.0f) {
        llama_sampler_chain_add(data->sampler, llama_sampler_init_typical(typical_p, 1));
    }

    // Top-P sampling
    if (top_p < 1.0f) {
        llama_sampler_chain_add(data->sampler, llama_sampler_init_top_p(top_p, 1));
    }

    // Min-P sampling
    if (min_p > 0.0f) {
        llama_sampler_chain_add(data->sampler, llama_sampler_init_min_p(min_p, 1));
    }

    // XTC sampling
    if (xtc_probability > 0.0f && xtc_threshold > 0.0f) {
        llama_sampler_chain_add(data->sampler,
            llama_sampler_init_xtc(xtc_probability, xtc_threshold, 1, seed));
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

    // Get chat template
    const char * tmpl = llama_model_chat_template(data->model, nullptr);

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
        std::string complete_prompt;
        for (size_t i = 0; i < data->chat_messages.size(); i++) {
            if (i > 0) complete_prompt += "\n";
            complete_prompt += data->chat_messages[i].content;
        }

        new_len = snprintf(data->formatted_chat.data(), data->formatted_chat.size(),
                         "%s", complete_prompt.c_str());
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

    // Get the COMPLETE prompt
    std::string complete_prompt(data->formatted_chat.begin(), data->formatted_chat.begin() + new_len);

    env->ReleaseStringUTFChars(prompt_jstr, prompt_cstr);

    // Tokenize the COMPLETE prompt
    const int n_prompt_tokens = -llama_tokenize(vocab, complete_prompt.c_str(), complete_prompt.size(),
                                              NULL, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    if (llama_tokenize(vocab, complete_prompt.c_str(), complete_prompt.size(),
                      prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        set_last_error("Failed to tokenize the prompt");
        return -1;
    }

    fprintf(stderr, "\n=== Token processing ===\n");
    fprintf(stderr, "Complete prompt tokens: %d\n", n_prompt_tokens);
    fprintf(stderr, "Previous tokens processed: %d\n", data->n_past);
    fprintf(stderr, "n_keep: %d\n", data->n_keep);

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
    int min_len = std::min((int)data->all_tokens.size(), n_prompt_tokens);

    // If n_keep is 0, don't reuse any cache
    if (data->n_keep == 0) {
        // Clear everything
        llama_kv_cache_clear(data->ctx);
        data->all_tokens.clear();
        data->n_past = 0;
        common_prefix = 0;
    } else {
        // Find common prefix
        for (int i = 0; i < min_len; i++) {
            if (data->all_tokens[i] == prompt_tokens[i]) {
                common_prefix++;
            } else {
                break;
            }
        }

        // Limit by n_keep if specified
        if (data->n_keep > 0) {
            common_prefix = std::min(common_prefix, data->n_keep);
        }

        // Remove KV cache entries beyond common prefix
        if (common_prefix < data->n_past) {
            llama_kv_cache_seq_rm(data->ctx, 0, common_prefix, -1);
            data->n_past = common_prefix;
        }
    }

    fprintf(stderr, "Common prefix: %d tokens\n", common_prefix);
    fprintf(stderr, "Tokens to process: %d\n", n_prompt_tokens - common_prefix);
    fprintf(stderr, "=== End token processing ===\n");

    // Update stored tokens
    data->all_tokens = prompt_tokens;

    // Process new tokens in batches
    int n_batch = llama_n_batch(data->ctx);
    int n_ctx = llama_n_ctx(data->ctx);
    std::string response;

    // Process only new tokens (from common_prefix to end)
    for (int i = common_prefix; i < n_prompt_tokens; i += n_batch) {
        int batch_size = std::min(n_batch, n_prompt_tokens - i);
        llama_batch batch = llama_batch_get_one(prompt_tokens.data() + i, batch_size);

        if (data->n_past + batch.n_tokens > n_ctx) {
            fprintf(stderr, "context size exceeded\n");
            set_last_error("Context size exceeded");
            return -1;
        }

        int ret = llama_decode(data->ctx, batch);
        if (ret != 0) {
            set_last_error("Failed to decode prompt batch, ret = " + std::to_string(ret));
            return -1;
        }

        data->n_past += batch.n_tokens;
    }

    // Now generate the response token by token
    llama_token new_token_id;
    int n_generated = 0;

    while (n_generated < n_predict) {
        // Sample the next token
        new_token_id = llama_sampler_sample(data->sampler, data->ctx, -1);

        // Is it an end of generation?
        if (!ignore_eos && llama_vocab_is_eog(vocab, new_token_id)) {
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

        // Add to all_tokens
        data->all_tokens.push_back(new_token_id);

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
        llama_batch batch = llama_batch_get_one(&new_token_id, 1);

        if (data->n_past + batch.n_tokens > n_ctx) {
            fprintf(stderr, "context size exceeded\n");
            set_last_error("Context size exceeded");
            break;
        }

        int ret = llama_decode(data->ctx, batch);
        if (ret != 0) {
            set_last_error("Failed to decode, ret = " + std::to_string(ret));
            break;
        }

        data->n_past += batch.n_tokens;
    }

    // Add the response to the messages
    data->chat_messages.push_back({"assistant", strdup(response.c_str())});

    fprintf(stderr, "\n=== Generation complete ===\n");
    fprintf(stderr, "Generated %d tokens\n", n_generated);
    fprintf(stderr, "Total tokens in context: %d\n", data->n_past);
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

    // Free existing chat messages
    free_chat_messages(data->chat_messages);

    // Clear formatted chat buffer
    data->formatted_chat.clear();

    // Clear token tracking based on n_keep
    if (data->n_keep == 0) {
        // Clear everything
        data->all_tokens.clear();
        data->n_past = 0;
        llama_kv_cache_clear(data->ctx);
    } else if (data->n_keep > 0 && data->n_past > data->n_keep) {
        // Keep only first n_keep tokens
        data->all_tokens.resize(data->n_keep);
        data->n_past = data->n_keep;
        // Clear KV cache beyond n_keep
        llama_kv_cache_seq_rm(data->ctx, 0, data->n_keep, -1);
    }
    // If n_keep is -1, keep everything (don't clear)
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif
