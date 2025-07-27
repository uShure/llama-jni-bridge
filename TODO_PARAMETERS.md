# TODO: Parameters Not Yet Applied

This document lists all parameters that are extracted from Java but not yet applied in the C++ implementation.

## InitParams TODOs

1. **rope_scale** - RoPE scaling factor needs to be applied to context
2. **cache_type_k_draft** - Draft model K cache type (when draft model support is added)
3. **cache_type_v_draft** - Draft model V cache type (when draft model support is added)
4. **device** - GPU device configuration needs new API approach

## GenerateParams TODOs

### Basic Processing
1. **escape** - Process escape sequences in input
2. **no_escape** - Override to disable escape processing

### Penalty Parameters
3. **penalize_nl** - Apply newline penalty to penalty sampler

### XTC Sampling
4. **xtc_min** - Minimum tokens for XTC sampling (when XTC sampler is available)

### Context Control
5. **no_context_shift** - Disable automatic context shifting when full

### Output Control
6. **n_probs** - Number of token probabilities to return per step

### Group Attention
7. **grp_attn_n** - Group attention factor
8. **grp_attn_w** - Group attention width

### Missing Samplers
9. **TFS (Tail-free sampling)** - No longer available in API, removed from chain

## Implementation Notes

These parameters are correctly extracted from Java but need implementation when:
- The corresponding llama.cpp API becomes available
- The feature is added to the sampler chain
- The context/model configuration supports it

Most of these are advanced features that don't affect basic functionality. The wrapper works correctly without them but could be enhanced when the APIs are available.
