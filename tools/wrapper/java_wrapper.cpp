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

// --- Globals for safe resource management ---
static std::mutex g_context_registry_mutex;
struct LlamaContextData;
static std::unordered_set<LlamaContextData*> g_active_contexts;

// --- JNI Lifecycle Hooks ---

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
    llama_backend_init();
    llama_numa_init(ggml_numa_strategy::GGML_NUMA_STRATEGY_DISABLED);
    return JNI_VERSION_1_8;
}

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

// --- Context Data Structure ---

struct LlamaContextData {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    llama_sampler * sampler = nullptr;
    common_params params;
    int n_ctx = 0;
};

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
    data->params.devices.clear();  // Will use default device

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
        llama_model_free(data->model);
        return 0;
    }

    // Initialize sampler
    auto sparams = llama_sampler_chain_default_params();
    data->sampler = llama_sampler_chain_init(sparams);

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

    if (data->sampler) llama_sampler_free(data->sampler);
    if (data->ctx) llama_free(data->ctx);
    if (data->model) llama_model_free(data->model);
    delete data;
}

JNIEXPORT jint JNICALL Java_org_llm_wrapper_LlamaCpp_llama_1generate
  (JNIEnv *env, jclass, jlong handle, jobject generateParams) {

    LlamaContextData* data = reinterpret_cast<LlamaContextData*>(handle);
    if (!is_valid_context(data)) return -1;
    if (!data || !data->ctx) return -1;

    // Extract generation parameters
    jstring prompt_jstr = getStringField(env, generateParams, "prompt");
    const char* prompt_cstr = nullptr;
    if (prompt_jstr) {
        prompt_cstr = env->GetStringUTFChars(prompt_jstr, nullptr);
    }
    if (!prompt_cstr) return -1;

    int n_predict = getIntField(env, generateParams, "nPredict");
    int top_k = getIntField(env, generateParams, "topK");
    float top_p = getFloatField(env, generateParams, "topP");
    float temp = getFloatField(env, generateParams, "temp");
    float repeat_penalty = getFloatField(env, generateParams, "repeatPenalty");
    int repeat_last_n = getIntField(env, generateParams, "repeatLastN");
    bool stream = getBooleanField(env, generateParams, "stream");
    int seed = getIntField(env, generateParams, "seed");

    // Update sampler parameters
    llama_sampler_reset(data->sampler);
    llama_sampler_free(data->sampler);

    // Recreate sampler with new parameters
    auto sparams = llama_sampler_chain_default_params();
    data->sampler = llama_sampler_chain_init(sparams);

    llama_sampler_chain_add(data->sampler, llama_sampler_init_penalties(
        repeat_last_n,    // penalty_last_n
        repeat_penalty,   // penalty_repeat
        0.0f,             // penalty_freq
        0.0f              // penalty_present
    ));
    llama_sampler_chain_add(data->sampler, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(data->sampler, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(data->sampler, llama_sampler_init_temp(temp));
    llama_sampler_chain_add(data->sampler, llama_sampler_init_dist(seed));

    // Tokenize prompt
    std::vector<llama_token> tokens_list;
    tokens_list.resize(data->n_ctx);
    // Get vocab from model first
    const struct llama_vocab * vocab = llama_model_get_vocab(data->model);
    int n_tokens = llama_tokenize(vocab, prompt_cstr, strlen(prompt_cstr), tokens_list.data(), tokens_list.size(), true, false);
    if (n_tokens > 0) {
        tokens_list.resize(n_tokens);
    }

    env->ReleaseStringUTFChars(prompt_jstr, prompt_cstr);

    if (n_tokens < 0) {
        std::cerr << "Failed to tokenize prompt" << std::endl;
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

    // Clear memory (KV cache) if needed
    llama_memory_t mem = llama_get_memory(data->ctx);
    llama_memory_clear(mem, 0); // Clear seq_id 0

    // Evaluate initial prompt
    llama_batch batch = llama_batch_init(data->params.n_batch, 0, 1);

    for (int i = 0; i < n_tokens; i++) {
        common_batch_add(batch, tokens_list[i], i, { 0 }, false);
    }

    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(data->ctx, batch) != 0) {
        std::cerr << "llama_decode() failed" << std::endl;
        llama_batch_free(batch);
        return -1;
    }

    int n_cur = batch.n_tokens;
    int n_decode = 0;

    // Generate tokens
    std::string generated_text;

    while (n_cur < data->n_ctx && n_decode < n_predict) {
        // Sample next token
        llama_token new_token_id = llama_sampler_sample(data->sampler, data->ctx, -1);

        // Check for EOS
        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        // Add token to the output
        char token_str_buf[128];
        int32_t n = llama_token_to_piece(vocab, new_token_id, token_str_buf, sizeof(token_str_buf), 0, false);
        if (n < 0) {
            n = 0;
        }
        std::string token_str(token_str_buf, n);
        generated_text += token_str;

        // Stream callback
        if (stream && tokenCallback) {
            jstring tokenJStr = env->NewStringUTF(token_str.c_str());
            jboolean shouldContinue = env->CallBooleanMethod(tokenCallback, testMethod, tokenJStr);
            env->DeleteLocalRef(tokenJStr);

            if (!shouldContinue) {
                break;
            }
        }

        // Prepare next batch
        common_batch_clear(batch);
        common_batch_add(batch, new_token_id, n_cur, { 0 }, true);

        n_cur++;
        n_decode++;

        if (llama_decode(data->ctx, batch) != 0) {
            std::cerr << "llama_decode() failed" << std::endl;
            break;
        }
    }

    llama_batch_free(batch);

    return 0;
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif
