#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include "llama.h"
#include "common.h"

// Simple test for JNI wrapper debugging
int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path> [prompt]" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string prompt = argc > 2 ? argv[2] : "The capital of France is";

    std::cout << "=== JNI Wrapper Test ===" << std::endl;
    std::cout << "Model: " << model_path << std::endl;
    std::cout << "Prompt: " << prompt << std::endl;

    // Initialize backend
    std::cout << "\n1. Initializing backend..." << std::endl;
    llama_backend_init();
    llama_numa_init(ggml_numa_strategy::GGML_NUMA_STRATEGY_DISABLED);

    // Load model
    std::cout << "\n2. Loading model..." << std::endl;
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0; // CPU only for debugging

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        std::cerr << "Failed to load model!" << std::endl;
        llama_backend_free();
        return 1;
    }
    std::cout << "Model loaded successfully" << std::endl;

    // Create context
    std::cout << "\n3. Creating context..." << std::endl;
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    ctx_params.n_batch = 512;
    ctx_params.n_threads = 4;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        std::cerr << "Failed to create context!" << std::endl;
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    std::cout << "Context created successfully" << std::endl;

    // Initialize sampler
    std::cout << "\n4. Creating sampler..." << std::endl;
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * sampler = llama_sampler_chain_init(sparams);

    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(
        64,       // penalty_last_n
        1.1f,     // penalty_repeat
        0.0f,     // penalty_freq
        0.0f      // penalty_present
    ));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(12345));
    std::cout << "Sampler created" << std::endl;

    // Tokenize prompt
    std::cout << "\n5. Tokenizing prompt..." << std::endl;
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens;
    tokens.resize(512);

    int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(), tokens.data(), tokens.size(), true, false);
    if (n_tokens < 0) {
        std::cerr << "Tokenization failed!" << std::endl;
        llama_sampler_free(sampler);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    tokens.resize(n_tokens);
    std::cout << "Tokens: " << n_tokens << std::endl;

    // Clear memory
    std::cout << "\n6. Clearing memory..." << std::endl;
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_clear(mem, 0);
    std::cout << "Memory cleared" << std::endl;

    // Create batch
    std::cout << "\n7. Evaluating prompt..." << std::endl;
    llama_batch batch = llama_batch_init(512, 0, 1);

    for (int i = 0; i < n_tokens; i++) {
        common_batch_add(batch, tokens[i], i, { 0 }, false);
    }
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch) != 0) {
        std::cerr << "llama_decode() failed!" << std::endl;
        llama_batch_free(batch);
        llama_sampler_free(sampler);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    std::cout << "Prompt evaluated" << std::endl;

    // Generate
    std::cout << "\n8. Generating text..." << std::endl;
    std::cout << prompt;

    int n_cur = batch.n_tokens;
    int n_gen = 50; // Generate 50 tokens

    for (int i = 0; i < n_gen; i++) {
        llama_token new_token_id = llama_sampler_sample(sampler, ctx, -1);

        if (llama_vocab_is_eog(vocab, new_token_id)) {
            std::cout << " [EOS]" << std::endl;
            break;
        }

        // Convert token to string
        char token_str_buf[128];
        int32_t n = llama_token_to_piece(vocab, new_token_id, token_str_buf, sizeof(token_str_buf), 0, false);
        if (n > 0) {
            std::string token_str(token_str_buf, n);
            std::cout << token_str << std::flush;
        }

        // Prepare next batch
        common_batch_clear(batch);
        common_batch_add(batch, new_token_id, n_cur, { 0 }, true);
        n_cur++;

        if (llama_decode(ctx, batch) != 0) {
            std::cerr << "\nllama_decode() failed during generation!" << std::endl;
            break;
        }
    }
    std::cout << std::endl;

    // Cleanup
    std::cout << "\n9. Cleaning up..." << std::endl;

    std::cout << "  - Freeing batch..." << std::endl;
    llama_batch_free(batch);

    std::cout << "  - Freeing sampler..." << std::endl;
    llama_sampler_free(sampler);

    std::cout << "  - Freeing context..." << std::endl;
    llama_free(ctx);

    std::cout << "  - Freeing model..." << std::endl;
    llama_model_free(model);

    std::cout << "  - Freeing backend..." << std::endl;
    llama_backend_free();

    std::cout << "\n✓ Test completed successfully!" << std::endl;

    return 0;
}
