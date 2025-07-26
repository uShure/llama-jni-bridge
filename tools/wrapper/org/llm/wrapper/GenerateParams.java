package org.llm.wrapper;

import java.util.function.Predicate;

public class GenerateParams {
    public String prompt;
    public boolean useConversation = false;
    public boolean addAssistant = true;
    public int nThreads = 4;
    public int nBatch = 512;
    public int nPredict = 512;
    public int topK = 40;
    public float topP = 0.95f;
    public float temp = 0.8f;
    public float repeatPenalty = 1.1f;
    public int repeatLastN = 64;
    // stream parameter removed - streaming is controlled by tokenCallback presence
    public Predicate<String> tokenCallback;
    public int seed = -1;

    // Token retention control
    public int keep = -1;  // -1 = keep all tokens, 0 = disable caching, N = keep N tokens
}
