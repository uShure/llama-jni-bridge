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

// Global flag to track initialization
static bool g_llama_initialized = false;
static std::mutex g_init_mutex;

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM * /* vm */, void * /* reserved */) {
    std::lock_guard<std::mutex> lock(g_init_mutex);

    if (!g_llama_initialized) {
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
        g_llama_initialized = true;
    }

    last_error_message = "No error";
    return JNI_VERSION_1_8;
}

JNIEXPORT void JNICALL JNI_OnUnload(JavaVM * /* vm */, void * /* reserved */) {
    std::lock_guard<std::mutex> lock1(g_init_mutex);
    std::lock_guard<std::mutex> lock2(g_context_registry_mutex);

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

    if (g_llama_initialized) {
        llama_backend_free();
        g_llama_initialized = false;
    }
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
  (JNIEnv *env, jclass, jobject modelParams, jobject computeParams) {
    auto data = std::make_unique<LlamaContextData>();

    // Extract model path
    const char* modelPath_cstr = nullptr;
    jstring modelPath_jstr = getStringField(env, modelParams, "model");
    if (modelPath_jstr) {
        modelPath_cstr = env->GetStringUTFChars(modelPath_jstr, nullptr);
    }

    if (!modelPath_cstr) {
        set_last_error("Model path is required");
        return 0;
    }

    // Extract model parameters
    int n_gpu_layers = getIntField(env, modelParams, "gpuLayers");
    int main_gpu = getIntField(env, modelParams, "mainGpu");
    bool use_mlock = getBooleanField(env, modelParams, "mlock");
    bool no_mmap = getBooleanField(env, modelParams, "noMmap");
    bool check_tensors = getBooleanField(env, modelParams, "checkTensors");

    // Context parameters
    data->params.n_ctx = getIntField(env, modelParams, "ctxSize");
    data->params.n_batch = getIntField(env, modelParams, "batchSize");
    int ubatch_size = getIntField(env, modelParams, "ubatchSize");
    bool flash_attn = getBooleanField(env, modelParams, "flashAttn");
    bool no_kv_offload = getBooleanField(env, modelParams, "noKvOffload");

    // Compute parameters (if provided)
    if (computeParams != nullptr) {
        data->params.cpuparams.n_threads = getIntField(env, computeParams, "threads");
        data->params.cpuparams_batch.n_threads = getIntField(env, computeParams, "threadsBatch");

        // Apply thread defaults
        if (data->params.cpuparams.n_threads == -1) {
            data->params.cpuparams.n_threads = std::thread::hardware_concurrency();
        }
        if (data->params.cpuparams_batch.n_threads == -1) {
            data->params.cpuparams_batch.n_threads = data->params.cpuparams.n_threads;
        }
    } else {
        // Use default thread count
        int default_threads = std::thread::hardware_concurrency();
        data->params.cpuparams.n_threads = default_threads;
        data->params.cpuparams_batch.n_threads = default_threads;
    }

    data->params.devices.clear();  // Will use default device

    // Match simple-chat.cpp: if n_batch is 0, set it to n_ctx
    if (data->params.n_batch == 0) {
        data->params.n_batch = data->params.n_ctx;
    }

    data->n_ctx = data->params.n_ctx;

    // Load model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    model_params.main_gpu = main_gpu;
    model_params.use_mmap = !no_mmap;  // Note: inverted logic
    model_params.use_mlock = use_mlock;
    model_params.check_tensors = check_tensors;

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
    ctx_params.n_ubatch = ubatch_size;
    ctx_params.n_threads = data->params.cpuparams.n_threads;
    ctx_params.n_threads_batch = data->params.cpuparams_batch.n_threads;
    ctx_params.flash_attn = flash_attn;
    ctx_params.no_perf = false;  // Allow performance metrics
    ctx_params.embeddings = false;  // Not in new params

    data->ctx = llama_init_from_model(data->model, ctx_params);
    if (!data->ctx) {
        std::cerr << "Failed to create context" << std::endl;
        set_last_error("Failed to create context from model");
        llama_model_free(data->model);
        return 0;
    }

    // Initialize default sampler (will be recreated in generate with user params)
    data->sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(data->sampler, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(data->sampler, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(data->sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

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

    // Free chat messages
    for (auto & msg : data->chat_messages) {
        free(const_cast<char *>(msg.content));
    }
    data->chat_messages.clear();

    if (data->sampler) llama_sampler_free(data->sampler);
    if (data->ctx) llama_free(data->ctx);
    if (data->model) llama_model_free(data->model);
    delete data;
}

JNIEXPORT jint JNICALL Java_org_llm_wrapper_LlamaCpp_llama_1generate
  (JNIEnv *env, jclass, jlong handle, jobject samplingParams, jobject tokenCallback) {

    LlamaContextData* data = reinterpret_cast<LlamaContextData*>(handle);
    if (!is_valid_context(data)) {
        set_last_error("Invalid context handle");
        return -1;
    }
    if (!data || !data->ctx) {
        set_last_error("Context is null or not initialized");
        return -1;
    }

    // Extract prompt
    jstring prompt_jstr = getStringField(env, samplingParams, "prompt");
    const char* prompt_cstr = nullptr;
    if (prompt_jstr) {
        prompt_cstr = env->GetStringUTFChars(prompt_jstr, nullptr);
    }
    if (!prompt_cstr) {
        set_last_error("Prompt is required");
        return -1;
    }

    std::string user_input(prompt_cstr);

    // Extract sampling parameters
    int n_predict = getIntField(env, samplingParams, "predict");
    int keep = getIntField(env, samplingParams, "keep");
    int seed = getIntField(env, samplingParams, "seed");
    bool ignore_eos = getBooleanField(env, samplingParams, "ignoreEos");
    bool chat_mode = getBooleanField(env, samplingParams, "chatMode");
    bool stream = getBooleanField(env, samplingParams, "stream");

    // Temperature and sampling parameters
    float temp = getFloatField(env, samplingParams, "temp");
    int top_k = getIntField(env, samplingParams, "topK");
    float top_p = getFloatField(env, samplingParams, "topP");
    float min_p = getFloatField(env, samplingParams, "minP");
    float typical = getFloatField(env, samplingParams, "typical");

    // Repetition penalties
    int repeat_last_n = getIntField(env, samplingParams, "repeatLastN");
    float repeat_penalty = getFloatField(env, samplingParams, "repeatPenalty");
    float presence_penalty = getFloatField(env, samplingParams, "presencePenalty");
    float frequency_penalty = getFloatField(env, samplingParams, "frequencyPenalty");

    // Get vocab from model
    const struct llama_vocab * vocab = llama_model_get_vocab(data->model);

    // Check if this is first prompt in the conversation
    const bool is_first = llama_memory_seq_pos_max(llama_get_memory(data->ctx), 0) == -1;

    // Log generation parameters for debugging
    fprintf(stderr, "\n=== Generation parameters ===\n");
    fprintf(stderr, "n_predict: %d\n", n_predict);
    fprintf(stderr, "chat_mode: %s\n", chat_mode ? "true" : "false");
    fprintf(stderr, "temp: %.2f\n", temp);
    fprintf(stderr, "top_k: %d\n", top_k);
    fprintf(stderr, "top_p: %.2f\n", top_p);
    fprintf(stderr, "min_p: %.2f\n", min_p);
    fprintf(stderr, "typical: %.2f\n", typical);
    fprintf(stderr, "repeat_penalty: %.2f\n", repeat_penalty);
    fprintf(stderr, "repeat_last_n: %d\n", repeat_last_n);
    fprintf(stderr, "presence_penalty: %.2f\n", presence_penalty);
    fprintf(stderr, "frequency_penalty: %.2f\n", frequency_penalty);
    fprintf(stderr, "seed: %d\n", seed);
    fprintf(stderr, "stream: %s\n", stream ? "true" : "false");
    fprintf(stderr, "ignore_eos: %s\n", ignore_eos ? "true" : "false");
    fprintf(stderr, "=== End parameters ===\n");

    // Reset sampler for new generation
    llama_sampler_reset(data->sampler);

    // Recreate sampler chain with user-provided parameters
    llama_sampler_free(data->sampler);
    data->sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());

    // Add samplers based on user parameters
    // The order matches the default sampler sequence
    if (repeat_penalty != 1.0f || presence_penalty != 0.0f || frequency_penalty != 0.0f) {
        llama_sampler_chain_add(data->sampler, llama_sampler_init_penalties(
            repeat_last_n,       // penalty_last_n
            repeat_penalty,      // penalty_repeat
            frequency_penalty,   // penalty_freq
            presence_penalty     // penalty_present
        ));
    }

    if (top_k > 0 && top_k < llama_vocab_n_tokens(vocab)) {
        llama_sampler_chain_add(data->sampler, llama_sampler_init_top_k(top_k));
    }

    if (typical < 1.0f && typical > 0.0f) {
        llama_sampler_chain_add(data->sampler, llama_sampler_init_typical(typical, 1));
    }

    if (top_p < 1.0f && top_p > 0.0f) {
        llama_sampler_chain_add(data->sampler, llama_sampler_init_top_p(top_p, 1));
    }

    if (min_p > 0.0f) {
        llama_sampler_chain_add(data->sampler, llama_sampler_init_min_p(min_p, 1));
    }

    llama_sampler_chain_add(data->sampler, llama_sampler_init_temp(temp));
    llama_sampler_chain_add(data->sampler, llama_sampler_init_dist(seed >= 0 ? seed : LLAMA_DEFAULT_SEED));

    // Handle plain text mode vs chat mode
    std::string formatted_prompt;

    if (!chat_mode) {
        // Plain text mode - use prompt directly without chat template
        formatted_prompt = user_input;

        // For KV cache reuse in plain text mode, check if this prompt
        // starts with the previous prompt
        if (data->prev_len > 0 && formatted_prompt.length() >= data->prev_len) {
            std::string prev_prompt(data->formatted_chat.data(), data->prev_len);
            if (formatted_prompt.substr(0, data->prev_len) == prev_prompt) {
                fprintf(stderr, "Plain text mode: reusing KV cache for %d tokens\n", data->prev_len);
                // The KV cache from previous generation can be reused
            } else {
                fprintf(stderr, "Plain text mode: clearing KV cache (prompt changed)\n");
                llama_kv_cache_clear(data->ctx);
                data->prev_len = 0;
            }
        }
    } else {
        // Chat mode - use chat template
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
        formatted_prompt = std::string(data->formatted_chat.begin() + data->prev_len,
                                      data->formatted_chat.begin() + new_len);

        // Update prev_len for next iteration
        data->prev_len = new_len;
    }

    // For plain text mode, save the formatted prompt for next KV cache check
    if (!chat_mode) {
        if (formatted_prompt.size() > data->formatted_chat.size()) {
            data->formatted_chat.resize(formatted_prompt.size());
        }
        std::copy(formatted_prompt.begin(), formatted_prompt.end(), data->formatted_chat.begin());
        data->prev_len = formatted_prompt.size();
    }

    // Log the formatted prompt for debugging
    fprintf(stderr, "\n=== Debug: Prompt formatting ===\n");
    fprintf(stderr, "Mode: %s\n", chat_mode ? "chat" : "plain text");
    fprintf(stderr, "Previous length: %d\n", data->prev_len);
    fprintf(stderr, "\n=== Formatted prompt ===\n");
    fprintf(stderr, "%s", formatted_prompt.c_str());
    fprintf(stderr, "\n=== End of formatted prompt (length: %zu) ===\n", formatted_prompt.length());

    // Log first few tokens for debugging
    if (formatted_prompt.length() > 0) {
        fprintf(stderr, "First 10 chars (hex): ");
        for (size_t i = 0; i < std::min(formatted_prompt.length(), size_t(10)); i++) {
            fprintf(stderr, "%02x ", (unsigned char)formatted_prompt[i]);
        }
        fprintf(stderr, "\n");
    }

    env->ReleaseStringUTFChars(prompt_jstr, prompt_cstr);

    // Tokenize the prompt
    // In plain text mode, we may need to tokenize only the new part if reusing KV cache
    std::string prompt_to_tokenize = formatted_prompt;
    bool add_bos = is_first;

    if (!chat_mode && data->prev_len > 0 && formatted_prompt.length() > data->prev_len) {
        // In plain text mode with KV cache reuse, tokenize only the new part
        prompt_to_tokenize = formatted_prompt.substr(data->prev_len);
        add_bos = false;  // Don't add BOS for continuation
        fprintf(stderr, "Tokenizing only new part: %zu chars\n", prompt_to_tokenize.length());
    }

    const int n_prompt_tokens = -llama_tokenize(vocab, prompt_to_tokenize.c_str(), prompt_to_tokenize.size(),
                                              NULL, 0, add_bos, true);
    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    if (llama_tokenize(vocab, prompt_to_tokenize.c_str(), prompt_to_tokenize.size(),
                      prompt_tokens.data(), prompt_tokens.size(), add_bos, true) < 0) {
        set_last_error("Failed to tokenize the prompt");
        return -1;
    }

    // Setup token callback if provided
    jclass predicateClass = nullptr;
    jmethodID testMethod = nullptr;

    if (tokenCallback != nullptr) {
        predicateClass = env->GetObjectClass(tokenCallback);
        testMethod = env->GetMethodID(predicateClass, "test", "(Ljava/lang/Object;)Z");
    }

    // Process prompt tokens in batches (matching simple-chat.cpp)
    int n_batch = llama_n_batch(data->ctx);
    int n_ctx = llama_n_ctx(data->ctx);
    std::string response;

    fprintf(stderr, "\n=== Batch processing ===\n");
    fprintf(stderr, "Prompt tokens: %d\n", n_prompt_tokens);
    fprintf(stderr, "Batch size: %d\n", n_batch);
    fprintf(stderr, "Context size: %d\n", n_ctx);

    // First, process the prompt tokens in batches
    for (int i = 0; i < n_prompt_tokens; i += n_batch) {
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

    while (n_predict == -1 || n_generated < n_predict) {
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

    // Handle response based on mode
    if (chat_mode) {
        // Add the response to the messages for chat mode
        data->chat_messages.push_back({"assistant", strdup(response.c_str())});

        // Update prev_len to point to end of all messages (without adding assistant prefix)
        const char * tmpl = llama_model_chat_template(data->model, nullptr);
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
    } else {
        // In plain text mode, update the stored prompt with the full generated text
        std::string full_text = formatted_prompt + response;
        if (full_text.size() > data->formatted_chat.size()) {
            data->formatted_chat.resize(full_text.size());
        }
        std::copy(full_text.begin(), full_text.end(), data->formatted_chat.begin());
        data->prev_len = full_text.size();
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

    // Clear the KV cache to start fresh
    llama_memory_clear(llama_get_memory(data->ctx), true);
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif
