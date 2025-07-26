package org.llm.wrapper;

public class InitParams {
    // Model path
    public String modelPath;

    // Context parameters
    public int nCtx = 2048;              // Context size
    public int nBatch = 512;             // Batch size for prompt processing
    public int nUBatch = 512;            // Batch size for generation
    public int nSeqMax = 1;              // Max number of sequences
    public int nThreads = 4;             // Number of threads for prompt processing
    public int nThreadsBatch = 4;        // Number of threads for batch processing

    // Model loading parameters
    public int nGpuLayers = 0;           // Number of layers to offload to GPU
    public boolean useMmap = true;       // Use memory mapping for model
    public boolean useMlock = false;     // Lock model in memory

    // Sampling parameters
    public int seed = -1;                // RNG seed (-1 for random)

    // Mode parameters
    public boolean embeddings = false;   // Enable embeddings mode
    public boolean loraBase = false;     // Model is LoRA base

    // Memory and performance
    public boolean flashAttn = true;     // Use flash attention
    public boolean noKVOffload = false;  // Disable KV cache offloading
    public boolean useNnApi = false;     // Use Android NNAPI

    // Token keeping parameter for cache reuse
    public int nKeep = 0;  // 0 = disabled, -1 = keep all, N = keep first N tokens

    // GPU device parameters
    public int[] gpuDevices = null;      // GPU device indices (null = use default)
    public float[] tensorSplits = null;  // Tensor split ratios per GPU
    public int mainGpu = 0;              // Main GPU for small tensors
    public int nGqa = 0;                 // Number of GQA groups
    public float ropeFreqBase = 0.0f;   // RoPE frequency base
    public float ropeFreqScale = 0.0f;  // RoPE frequency scale
    public float yarnExtFactor = -1.0f; // YaRN extrapolation factor
    public float yarnAttnFactor = 1.0f; // YaRN attention factor
    public float yarnBetaFast = 32.0f;  // YaRN beta fast
    public float yarnBetaSlow = 1.0f;   // YaRN beta slow
    public int yarnOrigCtx = 0;         // YaRN original context size

    // Cache type parameters
    public int cacheTypeK = 1;           // GGML_TYPE_F16 by default
    public int cacheTypeV = 1;           // GGML_TYPE_F16 by default
    public int cacheTypeKDraft = 1;      // For draft models
    public int cacheTypeVDraft = 1;      // For draft models

    // Quantization parameters
    public boolean quantizeOutputTensor = false;  // Quantize output tensor
    public boolean onlyWriteModelTensor = false;  // Only write model tensor

    // Defragmentation
    public float defragThreshold = -1.0f; // KV cache defragmentation threshold

    // Other parameters
    public boolean noEscape = false;      // Disable escape sequences in output
    public boolean numa = false;          // Enable NUMA optimizations
    public int splitMode = 1;             // Split mode (1 = layer, 2 = row)
    public String tensorSplitStr = "";    // String representation of tensor splits
    public String loraPath = "";          // Path to LoRA adapter
    public float loraScale = 1.0f;        // LoRA adapter scale
    public String draftModel = "";        // Path to draft model for speculative
    public int nDraft = 8;                // Number of tokens to draft
    public int nChunks = -1;              // Max number of chunks
    public int nParallel = 1;             // Number of parallel sequences
    public int nSequences = 1;            // Number of sequences to generate
    public float pSplit = 0.1f;           // Speculative decoding split probability

    // Batching parameters
    public int nBatchMin = 0;             // Minimum batch size
    public int nBatchMax = 0;             // Maximum batch size

    // Type-K/V override
    public String typeK = "";             // Override type-k
    public String typeV = "";             // Override type-v

    // Logging
    public boolean logDisable = false;    // Disable logging
}
