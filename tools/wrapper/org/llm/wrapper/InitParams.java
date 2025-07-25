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
}