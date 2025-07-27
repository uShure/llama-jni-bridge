package org.llm.wrapper;

import java.util.function.Predicate;

/**
 * Complete generation parameters matching LLMParams
 */
public class GenerateParams {
    // === Required Parameters ===
    public String prompt;                         // Input prompt
    public int nPredict = 128;                   // Tokens to predict (default: 128, -1=inf, -2=fill context)
    public String file = "";                     // File containing prompt
    public String binaryFile = "";               // Binary file containing prompt

    // === Escape Processing ===
    public boolean escape = true;                // Process escape sequences
    public boolean noEscape = false;             // Override to disable escapes

    // === Basic Sampling Parameters ===
    public float temp = 0.8f;                    // Temperature
    public int topK = 40;                        // Top-K sampling (0 = disabled)
    public float topP = 0.9f;                    // Top-P sampling (1.0 = disabled)
    public float minP = 0.1f;                    // Min-P sampling (0.0 = disabled)
    public float tfsZ = 1.0f;                    // Tail-free sampling (1.0 = disabled)
    public float typicalP = 1.0f;                // Typical-P sampling (1.0 = disabled)
    public float topNsigma = -1.0f;              // Top-n-sigma sampling (-1.0 = disabled)

    // === Sampling Control ===
    public String samplers = "penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature";
    public String samplingSeq = "edskypmxt";     // Simplified sampler sequence
    public int seed = -1;                        // Sampling seed (-1 = random)

    // === Repetition Penalty Parameters ===
    public float repeatPenalty = 1.0f;           // Repetition penalty (1.0 = disabled)
    public int repeatLastN = 64;                 // Last N tokens for penalty (0 = disabled, -1 = ctx_size)
    public float frequencyPenalty = 0.0f;        // Frequency penalty (0.0 = disabled)
    public float presencePenalty = 0.0f;         // Presence penalty (0.0 = disabled)
    public boolean penalizeNl = false;           // Penalize newlines

    // === Mirostat Sampling ===
    public int mirostat = 0;                     // 0 = disabled, 1 = v1, 2 = v2
    public float mirostatTau = 5.0f;             // Target entropy (tau)
    public float mirostatEta = 0.1f;             // Learning rate (eta)

    // === DRY (Don't Repeat Yourself) Sampling ===
    public float dryMultiplier = 0.0f;           // DRY multiplier (0.0 = disabled)
    public float dryBase = 1.75f;                // DRY base value
    public int dryAllowedLength = 2;             // Allowed repetition length
    public int dryPenaltyLastN = -1;             // Last N tokens (0 = disabled, -1 = context size)
    public String[] drySequenceBreakers = null;  // Custom sequence breakers (null = use defaults)

    // === XTC (eXtended Temperature Control) Sampling ===
    public float xtcThreshold = 0.1f;            // XTC threshold (1.0 = disabled)
    public float xtcProbability = 0.0f;          // XTC probability (0.0 = disabled)
    public int xtcMin = 0;                       // XTC minimum tokens
    public boolean xtcProbabilisticThreshold = false;

    // === Dynamic Temperature ===
    public float dynTempRange = 0.0f;            // Dynamic temp range (0.0 = disabled)
    public float dynTempExponent = 1.0f;         // Dynamic temp exponent

    // === Grammar and Constraints ===
    public String grammar = "";                  // BNF-like grammar
    public String grammarFile = "";              // Grammar file path
    public String jsonSchema = "";               // JSON schema constraint
    public String jsonSchemaFile = "";           // JSON schema file path

    // === Logit Bias ===
    public String logitBias = "";                // TOKEN_ID(+/-)BIAS pairs

    // === Token Control ===
    public String[] stopSequences = null;        // Stop generation sequences
    public boolean ignoreEos = false;            // Ignore end-of-sequence token
    public boolean noContextShift = false;       // Disable context shift
    public boolean addBos = false;               // Add beginning-of-sequence
    public boolean noPenalizeEos = false;        // Don't penalize EOS

    // === Output Control ===
    public int keep = -1;                        // Tokens to keep (-1 = all, 0 = disable, N = keep N)
    public int nProbs = 0;                       // Number of probabilities to return
    public int minKeep = 0;                      // Minimum tokens to keep
    public boolean noDisplayPrompt = false;      // Don't display prompt
    public boolean noParseSpecialTokens = false; // Don't parse special tokens
    public boolean displayPrompt = true;         // Display the prompt
    public boolean verbose = false;              // Verbose output

    // === Conversation Mode ===
    public boolean conversation = false;         // Enable conversation mode
    public boolean chatTemplate = true;          // Use chat template
    public boolean multiline = false;            // Enable multiline input
    public boolean useConversation = false;      // Legacy conversation flag
    public boolean addAssistant = true;          // Add assistant message

    // === Plain Text Mode ===
    public boolean chatMode = true;              // true = chat with history/templates, false = plain text mode

    // === Special Tokens ===
    public String bosToken = "";                 // Beginning-of-sequence token
    public String eosToken = "";                 // End-of-sequence token

    // === Infill Mode ===
    public boolean infill = false;               // Enable infill mode
    public String inputPrefix = "";              // Infill prefix
    public String inputSuffix = "";              // Infill suffix

    // === Cache Control ===
    public boolean cachePrompt = true;           // Cache the prompt
    public String cacheSession = "";             // Session file for cache
    public boolean truncatePrompt = false;       // Truncate prompt if too long

    // === CFG Guidance ===
    public float guidanceScale = 1.0f;           // CFG guidance scale
    public String negativePrompt = "";           // Negative prompt for guidance

    // === Speculative Decoding ===
    public int nDraft = -1;                      // Number of draft tokens
    public float pAccept = 0.0f;                 // Acceptance probability
    public float pSplit = 0.0f;                  // Split probability

    // === Group Attention ===
    public int grpAttnN = 1;                     // Group-attention factor
    public int grpAttnW = 512;                   // Group-attention width

    // === Warmup ===
    public boolean noWarmup = false;             // Skip model warmup

    // === Thread Control (for generation) ===
    public int nThreads = -1;                    // Override threads for this generation
    public int nBatch = -1;                      // Override batch size for this generation

    // === Token Callback ===
    public Predicate<String> tokenCallback;      // Return false to stop generation

    // === Helper Methods ===

    /**
     * Set DRY sequence breakers from a single string
     */
    public void setDrySequenceBreaker(String breaker) {
        if ("none".equalsIgnoreCase(breaker)) {
            this.drySequenceBreakers = new String[0];
        } else {
            // Split by some delimiter if needed, or just use as single breaker
            this.drySequenceBreakers = new String[] { breaker };
        }
    }

    /**
     * Configure for maximum quality (slower)
     */
    public void setQualityPreset() {
        this.temp = 0.7f;
        this.topK = 50;
        this.topP = 0.95f;
        this.minP = 0.05f;
        this.repeatPenalty = 1.1f;
        this.repeatLastN = 256;
        this.dryMultiplier = 0.8f;
        this.dynTempRange = 0.5f;
    }

    /**
     * Configure for speed (lower quality)
     */
    public void setSpeedPreset() {
        this.temp = 0.8f;
        this.topK = 30;
        this.topP = 0.9f;
        this.minP = 0.1f;
        this.repeatPenalty = 1.05f;
        this.repeatLastN = 64;
        this.dryMultiplier = 0.0f;
        this.dynTempRange = 0.0f;
    }

    /**
     * Configure for deterministic output
     */
    public void setDeterministicPreset() {
        this.temp = 0.0f;
        this.topK = 1;
        this.seed = 42;
        this.dryMultiplier = 0.0f;
        this.mirostat = 0;
    }
}
