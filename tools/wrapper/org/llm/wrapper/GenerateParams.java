package org.llm.wrapper;

import java.util.function.Predicate;

public class GenerateParams {
    public String prompt;
    public int nThreads = 4;
    public int nBatch = 512;
    public int nPredict = 512;
    public int topK = 40;
    public float topP = 0.95f;
    public float temp = 0.8f;
    public float repeatPenalty = 1.1f;
    public int repeatLastN = 64;
    public boolean stream = true;
    public Predicate<String> tokenCallback;
    public int seed = -1;
}