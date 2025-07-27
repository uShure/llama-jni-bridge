# llama.cpp API Changes and Fixes

## Fixed Compilation Errors

### 1. Device configuration
- **Old**: `data->params.device = device_cstr;`
- **Fixed**: Removed - device configuration now handled through model/context params

### 2. LLAMA_MAX_DEVICES constant
- **Old**: `tensor_split_vec.size() <= LLAMA_MAX_DEVICES`
- **Fixed**: Removed check - constant no longer exists

### 3. llama_sampler_init_logit_bias parameters
- **Old**: `llama_sampler_init_logit_bias(n_vocab, 0, nullptr, nullptr)`
- **Fixed**: `llama_sampler_init_logit_bias(n_vocab, 0, nullptr)` - now takes 3 params

### 4. Tail-free sampling function
- **Old**: `llama_sampler_init_tail_free(tfs_z, 1)`
- **Attempted**: `llama_sampler_init_tfs(tfs_z)`
- **Fixed**: Removed - TFS sampling no longer available in API

### 5. KV cache functions (using deprecated API)
- **Old**: `llama_kv_cache_clear()`
- **Attempted**: `llama_memory_clear()`
- **Fixed**: `llama_kv_self_clear()` - using deprecated but working function

- **Old**: `llama_kv_cache_seq_rm()`
- **Attempted**: `llama_memory_seq_rm()`
- **Fixed**: `llama_kv_self_seq_rm()` - using deprecated but working function

### 6. Unused variables
- Added `(void)variable;` to suppress warnings for:
  - `rope_scale` - TODO: Apply when API supports it
  - `cache_type_k_draft` - TODO: Apply for draft models
  - `cache_type_v_draft` - TODO: Apply for draft models

## Build Instructions

After these fixes, the code should compile with newer versions of llama.cpp:

```bash
cd /path/to/llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
cmake --build build --config Release
```

## Notes

These changes reflect the evolution of the llama.cpp API:
- Some samplers like TFS have been removed
- Memory management API is in transition (old functions deprecated but still work)
- Device configuration moved to different structures
- Some parameters are extracted but not yet applied (marked with TODO)

The wrapper now uses deprecated but functional APIs where necessary to maintain compatibility.
