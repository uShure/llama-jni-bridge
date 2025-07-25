package org.llm.wrapper;

public class ModelParams {
    // Model loading parameters

    /** FNAME model path (maps to modelPath in context) */
    public String model;

    /** N number of layers to store in VRAM (maps to n_gpu_layers) */
    public int gpuLayers = 0;

    /** {none,layer,row} how to split the model across multiple GPUs */
    public String splitMode = "none";

    /** N0,N1,N2,... fraction of the model to offload to each GPU */
    public String tensorSplit;

    /** INDEX the GPU to use for the model */
    public int mainGpu = 0;

    /** FNAME path to LoRA adapter */
    public String lora;

    /** FNAME SCALE path to LoRA adapter with user defined scaling */
    public String loraScaled;

    /** FNAME add a control vector */
    public String controlVector;

    /** FNAME SCALE add a control vector with user defined scaling */
    public String controlVectorScaled;

    /** START END layer range to apply the control vector(s) to */
    public int controlVectorLayerRange;

    // Memory and performance parameters

    /** Force system to keep model in RAM (maps to use_mlock) */
    public boolean mlock = false;

    /** Do not memory-map model (maps to use_mmap) */
    public boolean noMmap = false;

    /** Check model tensor data for invalid values */
    public boolean checkTensors = false;

    // Context parameters

    /** N size of the prompt context (maps to n_ctx) */
    public int ctxSize = 4096;

    /** N logical maximum batch size (maps to n_batch) */
    public int batchSize = 2048;

    /** N physical maximum batch size (maps to n_ubatch) */
    public int ubatchSize = 512;

    /** Use full-size SWA cache */
    public boolean swaFull = false;

    /** Enable Flash Attention */
    public boolean flashAttn = false;

    /** Disable KV offload */
    public boolean noKvOffload = false;

    /** TYPE KV cache data type for K (f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1) */
    public String cacheTypeK = "f16";

    /** TYPE KV cache data type for V */
    public String cacheTypeV = "f16";

    /** N KV cache defragmentation threshold */
    public float defragThold = 0.1f;

    // NUMA optimization

    /** TYPE NUMA optimization (distribute, isolate, numactl) */
    public String numa;

    // RoPE parameters

    /** {none,linear,yarn} RoPE frequency scaling method */
    public String ropeScaling = "linear";

    /** N RoPE context scaling factor */
    public float ropeScale;

    /** N RoPE base frequency */
    public float ropeFreqBase;

    /** N RoPE frequency scaling factor */
    public float ropeFreqScale;

    /** N YaRN: original context size of model */
    public int yarnOrigCtx = 0;

    /** N YaRN: extrapolation mix factor */
    public float yarnExtFactor = -1f;

    /** N YaRN: scale sqrt(t) or attention magnitude */
    public float yarnAttnFactor = 1f;

    /** N YaRN: high correction dim or alpha */
    public float yarnBetaSlow = 1f;

    /** N YaRN: low correction dim or beta */
    public float yarnBetaFast = 32f;

    // Logging

    /** Log disable */
    public boolean logDisable = false;

    /** FNAME Log to file */
    public String logFile;

    /** Enable prefix in log messages */
    public boolean logPrefix = false;

    /** Enable timestamps in log messages */
    public boolean logTimestamps = false;

    // Advanced overrides

    /** KEY=TYPE:VALUE advanced option to override model metadata */
    public String overrideKv;

    /** Disable offloading host tensor operations to device */
    public boolean noOpOffload = false;

    /** <dev1,dev2,..> comma-separated list of devices to use for offloading */
    public String device;

    /** <tensor name pattern> override tensor buffer type */
    public String overrideTensor;
}
