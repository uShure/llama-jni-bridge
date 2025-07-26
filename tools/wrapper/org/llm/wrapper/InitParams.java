package org.llm.wrapper;

public class InitParams {
    public String modelPath;
    public int nCtx = 2048;
    public int nBatch = 512;
    public int nGpuLayers = 0;
    public int nThreads = 4;
    public int nThreadsBatch = 4;
    public int seed = -1;
    public boolean useMmap = true;
    public boolean useMlock = false;
    public boolean embeddings = false;

    // Token keeping parameter for cache reuse
    public int nKeep = 0;  // 0 = disabled, -1 = keep all, N = keep first N tokens

    // GPU device parameters
    public int[] gpuDevices = null; // null means use default device
    public float[] tensorSplits = null; // null means use default splits

    // Cache type parameters
    public int cacheTypeK = 1; // GGML_TYPE_F16 by default
    public int cacheTypeV = 1; // GGML_TYPE_F16 by default
    public int cacheTypeKDraft = 1; // For draft models
    public int cacheTypeVDraft = 1; // For draft models

    // Other missing parameters
    public boolean noEscape = false; // Disable escape sequences in output
}
