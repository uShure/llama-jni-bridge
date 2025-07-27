package org.llm.wrapper;

/**
 * Complete initialization parameters matching LLMParams
 */
public class InitParams {
    // Model path
    public String modelPath;

    // === Threading and CPU Configuration ===
    public int nThreads = -1;              // Number of threads for generation (-1 = auto)
    public int nThreadsBatch = -1;         // Number of threads for batch processing (-1 = same as nThreads)
    public String cpuMask = "";            // CPU affinity mask (hex)
    public String cpuRange = "";           // CPU range (e.g., "0-3")
    public boolean cpuStrict = false;      // Strict CPU placement
    public int priority = 0;               // Process priority: -1=low, 0=normal, 1=medium, 2=high, 3=realtime
    public int poll = 50;                  // Polling level (0-100, 0 = no polling)
    public String cpuMaskBatch = "";       // CPU mask for batch processing
    public String cpuRangeBatch = "";      // CPU range for batch processing
    public boolean cpuStrictBatch = false; // Strict CPU placement for batch
    public int priorityBatch = 0;          // Priority for batch processing
    public boolean pollBatch = true;       // Use polling for batch

    // === Context and Batch Configuration ===
    public int nCtx = 4096;                // Context size (0 = from model)
    public int nBatch = 2048;              // Logical batch size
    public int nUBatch = 512;              // Physical batch size
    public int nSeqMax = 1;                // Max number of sequences
    public int nKeep = 0;                  // Tokens to keep from initial prompt (0 = disabled, -1 = all)
    public boolean swaFull = false;        // Use full-size SWA cache

    // === Model Loading Parameters ===
    public boolean useMmap = true;         // Use memory mapping
    public boolean useMlock = false;       // Lock model in memory
    public boolean checkTensors = false;   // Check tensor data validity
    public boolean embeddings = false;     // Enable embeddings mode
    public boolean loraBase = false;       // Model is LoRA base

    // === GPU Configuration ===
    public int nGpuLayers = 0;             // Layers to offload to GPU
    public int splitMode = 0;              // 0=none, 1=layer, 2=row
    public String tensorSplit = "";        // Tensor split ratios (e.g., "3,1")
    public int mainGpu = -1;               // Main GPU index (-1 = default device)
    public String device = "";             // Comma-separated device list
    public String overrideTensor = "";     // Override tensor buffer types

    // === Performance Features ===
    public boolean flashAttn = false;      // Enable flash attention
    public boolean noKVOffload = false;    // Disable KV cache offloading
    public boolean noOpOffload = false;    // Disable operator offloading
    public boolean noPerf = false;         // Disable performance timings

    // === RoPE Configuration ===
    public int ropeScalingType = 1;        // 0=none, 1=linear, 2=yarn
    public float ropeScale = 0.0f;         // RoPE context scaling factor
    public float ropeFreqBase = 0.0f;      // RoPE base frequency
    public float ropeFreqScale = 0.0f;     // RoPE frequency scaling

    // === YaRN Parameters ===
    public int yarnOrigCtx = 0;            // Original context size
    public float yarnExtFactor = -1.0f;    // Extrapolation mix factor
    public float yarnAttnFactor = 1.0f;    // Attention magnitude scale
    public float yarnBetaSlow = 1.0f;      // High correction dim
    public float yarnBetaFast = 32.0f;     // Low correction dim

    // === Cache Configuration ===
    public int cacheTypeK = 1;             // KV cache type K (1 = f16)
    public int cacheTypeV = 1;             // KV cache type V (1 = f16)
    public int cacheTypeKDraft = 1;        // Draft model cache type K
    public int cacheTypeVDraft = 1;        // Draft model cache type V
    public float defragThreshold = 0.1f;   // KV cache defragmentation threshold

    // === NUMA Configuration ===
    public int numa = 0;                   // 0=none, 1=distribute, 2=isolate, 3=numactl

    // === LoRA and Control Vectors ===
    public String lora = "";               // Path to LoRA adapter
    public String loraScaled = "";         // Path to scaled LoRA adapter
    public float loraScale = 1.0f;         // LoRA adapter scale
    public String controlVector = "";      // Control vector path
    public String controlVectorScaled = "";// Scaled control vector
    public int controlVectorLayerStart = 0;// Control vector layer range start
    public int controlVectorLayerEnd = -1; // Control vector layer range end

    // === Sampling Configuration ===
    public int seed = -1;                  // RNG seed (-1 = random)

    // === Logging Configuration ===
    public boolean logDisable = false;     // Disable logging
    public String logFile = "";            // Log to file
    public boolean logPrefix = false;      // Enable log prefix
    public boolean logTimestamps = false;  // Enable log timestamps

    // === Draft Model ===
    public String draftModel = "";         // Path to draft model
    public int nDraft = 8;                 // Tokens to draft
    public float pSplit = 0.1f;            // Speculative decoding split probability

    // === Quantization ===
    public boolean quantizeOutputTensor = false;
    public boolean onlyWriteModelTensor = false;

    // === Model Metadata Overrides ===
    public String overrideKv = "";         // KEY=TYPE:VALUE overrides

    // === Backward Compatibility ===
    public boolean noEscape = false;       // Disable escape sequences
    public boolean useNnApi = false;       // Android NNAPI
    public int nGqa = 0;                   // Number of GQA groups
    public int nChunks = -1;               // Max chunks
    public int nParallel = 1;              // Parallel sequences
    public int nSequences = 1;             // Number of sequences
}
