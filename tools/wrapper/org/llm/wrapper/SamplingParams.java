package org.llm.wrapper;

public class SamplingParams {
    // Basic generation parameters

    /** PROMPT prompt to start generation with */
    public String prompt;

    /** N number of tokens to predict (maps to n_predict) */
    public int predict = -1;

    /** N number of tokens to keep from the initial prompt */
    public int keep = 0;

    /** SEED RNG seed (-1 for random seed) */
    public int seed = -1;

    /** Ignore end of stream token and continue generating */
    public boolean ignoreEos = false;

    /** Disable internal libllama performance timings */
    public boolean noPerf = false;

    /** Process escape sequences (\n, \r, \t, \', \", \\) */
    public boolean escape = true;

    /** Disable context shift on infinite text generation */
    public boolean noContextShift = false;

    /** Skip warming up the model with an empty run */
    public boolean noWarmup = false;

    // Sampling parameters

    /** SAMPLERS samplers that will be used for generation in order */
    public String samplers = "penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature";

    /** SEQUENCE simplified sequence for samplers */
    public String samplingSeq = "edskypmxt";

    /** N temperature */
    public float temp = 0.8f;

    /** N top-k sampling (0 = disabled) */
    public int topK = 40;

    /** N top-p sampling (1.0 = disabled) */
    public float topP = 0.9f;

    /** N min-p sampling (0.0 = disabled) */
    public float minP = 0.1f;

    /** N top-n-sigma sampling (-1.0 = disabled) */
    public float topNsigma = -1f;

    /** N xtc probability (0.0 = disabled) */
    public float xtcProbability = 0f;

    /** N xtc threshold (1.0 = disabled) */
    public float xtcThreshold = 0.1f;

    /** N locally typical sampling (1.0 = disabled) */
    public float typical = 1f;

    // Dynamic temperature

    /** N dynamic temperature range (0.0 = disabled) */
    public float dynatempRange = 0f;

    /** N dynamic temperature exponent */
    public float dynatempExp = 1f;

    // Repetition penalties

    /** N last n tokens to consider for penalize (0 = disabled, -1 = ctx_size) */
    public int repeatLastN = 64;

    /** N penalize repeat sequence of tokens (1.0 = disabled) */
    public float repeatPenalty = 1f;

    /** N repeat alpha presence penalty (0.0 = disabled) */
    public float presencePenalty = 0f;

    /** N repeat alpha frequency penalty (0.0 = disabled) */
    public float frequencyPenalty = 0f;

    // DRY sampling parameters

    /** N set DRY sampling multiplier (0.0 = disabled) */
    public float dryMultiplier = 0f;

    /** N set DRY sampling base value */
    public float dryBase = 1.75f;

    /** N set allowed length for DRY sampling */
    public int dryAllowedLength = 2;

    /** N set DRY penalty for the last n tokens (0 = disable, -1 = context size) */
    public int dryPenaltyLastN = -1;

    /** STRING sequence breaker for DRY sampling */
    public String drySequenceBreaker;

    // Mirostat sampling

    /** N use Mirostat sampling (0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0) */
    public int mirostat = 0;

    /** N Mirostat learning rate */
    public float mirostatLr = 0.1f;

    /** N Mirostat target entropy */
    public float mirostatEnt = 5f;

    // Advanced parameters

    /** TOKEN_ID(+/-)BIAS modifies likelihood of token appearing */
    public String logitBias;

    /** GRAMMAR BNF-like grammar to constrain generations */
    public String grammar;

    /** FNAME file to read grammar from */
    public String grammarFile;

    /** SCHEMA JSON schema to constrain generations */
    public String jsonSchema;

    /** FILE File containing a JSON schema */
    public String jsonSchemaFile;

    /** N group-attention factor */
    public int grpAttnN = 1;

    /** N group-attention width */
    public int grpAttnW = 512;

    /** N number of parallel sequences to decode */
    public int parallel = 1;

    // Application-specific parameters (not from LLMParams)

    /** Whether to stream tokens as they are generated */
    public boolean stream = true;

    /** Whether to use chat mode with templates or plain text mode */
    public boolean chatMode = true;
}
