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
};

// --- Globals for safe resource management ---
static std::mutex g_context_registry_mutex;
static std::unordered_set<LlamaContextData*> g_active_contexts;

// --- Error handling ---
static std::mutex g_error_mutex;
static std::string last_error_message;

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
        // Free chat messages
        for (auto & msg : data->chat_messages) {
            free(const_cast<char *>(msg.content));
        }
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

// --- JNI Implementations ---

JNIEXPORT jlong JNICALL Java_org_llm_wrapper_LlamaCpp_llama_1init
  (JNIEnv *env, jobject, jobject initParams) {
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
    data->params.devices.clear();  // Will use default device

    // Match simple-chat.cpp: if n_batch is 0, set it to n_ctx
    if (data->params.n_batch == 0) {
        data->params.n_batch = data->params.n_ctx;
    }

    int n_gpu_layers = getIntField(env, initParams, "nGpuLayers");
    int seed = getIntField(env, initParams, "seed");
    bool use_mmap = getBooleanField(env, initParams, "useMmap");
    bool use_mlock = getBooleanField(env, initParams, "useMlock");
    bool embeddings_mode = getBooleanField(env, initParams, "embeddings");

    data->params.sampling.seed = seed;

    data->n_ctx = data->params.n_ctx;

    // Load model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    model_params.use_mmap = use_mmap;
    model_params.use_mlock = use_mlock;

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

    data->ctx = llama_init_from_model(data->model, ctx_params);
    if (!data->ctx) {
        std::cerr << "Failed to create context" << std::endl;
        set_last_error("Failed to create context from model");
        llama_model_free(data->model);
        return 0;
    }

    // Initialize sampler with simple-chat.cpp parameters
    data->sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(data->sampler, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(data->sampler, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(data->sampler, llama_sampler_init_dist(seed >= 0 ? seed : LLAMA_DEFAULT_SEED));

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
  (JNIEnv *, jobject, jlong handle) {
    LlamaContextData* data = reinterpret_cast<LlamaContextData*>(handle);
    if (!is_valid_context(data)) return;

    {
        std::lock_guard<std::mutex> lock(g_context_registry_mutex);
        g_active_contexts.erase(data);
    }

    if (data->sampler) llama_sampler_free(data->sampler);
    if (data->ctx) llama_free(data->ctx);
    if (data->model) llama_model_free(data->model);
    delete data;
}

JNIEXPORT jint JNICALL Java_org_llm_wrapper_LlamaCpp_llama_1generate
  (JNIEnv *env, jobject, jlong handle, jobject generateParams) {

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

    // Reset sampler for new generation
    llama_sampler_reset(data->sampler);

    // Get chat template
    const char * tmpl = llama_model_chat_template(data->model, nullptr);

    // Add user message to chat history
    data->chat_messages.push_back({"user", strdup(user_input.c_str())});

    // Format messages with chat template
    if (data->formatted_chat.empty()) {
        data->formatted_chat.resize(data->n_ctx);
    }

    int new_len = llama_chat_apply_template(tmpl, data->chat_messages.data(),
                                           data->chat_messages.size(), true,
                                           data->formatted_chat.data(), data->formatted_chat.size());

    if (new_len > (int)data->formatted_chat.size()) {
        data->formatted_chat.resize(new_len);
        new_len = llama_chat_apply_template(tmpl, data->chat_messages.data(),
                                          data->chat_messages.size(), true,
                                          data->formatted_chat.data(), data->formatted_chat.size());
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
    fprintf(stderr, "\n=== Debug: Formatted prompt (new part only) ===\n");
    fprintf(stderr, "%s", prompt.c_str());
    fprintf(stderr, "\n=== End of formatted prompt ===\n");

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

    // Process prompt tokens in batches (matching simple-chat.cpp)
    int n_batch = llama_n_batch(data->ctx);
    std::string response;

    // First, process the prompt tokens in batches
    for (int i = 0; i < n_prompt_tokens; i += n_batch) {
        int batch_size = std::min(n_batch, n_prompt_tokens - i);
        llama_batch batch = llama_batch_get_one(prompt_tokens.data() + i, batch_size);

        // Check if we have enough space in the context
        int n_ctx = llama_n_ctx(data->ctx);
        int n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(data->ctx), 0) + 1;
        if (n_ctx_used + batch.n_tokens > n_ctx) {
            fprintf(stderr, "context size exceeded\n");
            set_last_error("Context size exceeded");
            return -1;
        }

        int ret = llama_decode(data->ctx, batch);
        if (ret != 0) {
            set_last_error("Failed to decode prompt batch, ret = " + std::to_string(ret));
            return -1;
        }
    }

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
    data->prev_len = llama_chat_apply_template(tmpl, data->chat_messages.data(),
                                              data->chat_messages.size(), false,
                                              nullptr, 0);
    if (data->prev_len < 0) {
        set_last_error("Failed to apply the chat template");
        return -1;
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

    return static_cast<jlong>(model_size + ctx_size);
}

JNIEXPORT void JNICALL Java_org_llm_wrapper_LlamaCpp_llama_1clear_1chat
  (JNIEnv *, jclass, jlong handle) {
    LlamaContextData* data = reinterpret_cast<LlamaContextData*>(handle);
    if (!is_valid_context(data)) {
        return;
    }

    // Free existing chat messages
    for (auto & msg : data->chat_messages) {
        free(const_cast<char *>(msg.content));
    }
    data->chat_messages.clear();

    // Clear formatted chat buffer
    data->formatted_chat.clear();
    data->prev_len = 0;

    // Clear KV cache
    if (data->ctx) {
        llama_kv_cache_clear(data->ctx);
    }
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif
