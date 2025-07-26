package org.llm.wrapper;

import java.util.function.Predicate;

public class GenerateParams {
    // Required parameters
    public String prompt;                         // Input prompt
    public int nPredict = 128;                   // Max tokens to generate

    // Sampling parameters
    public float temp = 0.8f;                    // Temperature
    public int topK = 40;                        // Top-K sampling
    public float topP = 0.95f;                   // Top-P (nucleus) sampling
    public float minP = 0.05f;                   // Min-P sampling
    public float tfsZ = 1.0f;                    // Tail-free sampling
    public float typicalP = 1.0f;                // Typical sampling
    public int seed = -1;                        // Sampling seed (-1 for random)

    // Repetition penalty parameters
    public float repeatPenalty = 1.1f;           // Repetition penalty
    public int repeatLastN = 64;                 // Last N tokens for penalty
    public float frequencyPenalty = 0.0f;        // Frequency penalty
    public float presencePenalty = 0.0f;         // Presence penalty
    public boolean penalizeNl = false;           // Penalize newlines

    // Mirostat sampling
    public int mirostat = 0;                     // 0 = disabled, 1 = v1, 2 = v2
    public float mirostatTau = 5.0f;             // Target entropy
    public float mirostatEta = 0.1f;             // Learning rate

    // Token control
    public String[] stopSequences = null;        // Stop generation sequences
    public int keep = -1;                        // Tokens to keep (-1 = all, 0 = disable, N = keep N)
    public int nProbs = 0;                       // Number of probabilities to return
    public int minKeep = 0;                      // Minimum tokens to keep

    // Grammar
    public String grammar = "";                  // Grammar string
    public String grammarFile = "";              // Grammar file path

    // DRY (Don't Repeat Yourself) sampling
    public float dryMultiplier = 0.0f;          // DRY multiplier
    public float dryBase = 1.75f;               // DRY base
    public int dryAllowedLength = 2;            // Allowed length
    public int dryPenaltyLastN = -1;            // Last N tokens
    public String[] drySequenceBreakers = null; // Sequence breakers

    // XTC (eXtended Temperature Control) sampling
    public float xtcThreshold = 0.1f;           // XTC threshold
    public float xtcProbability = 0.0f;         // XTC probability
    public int xtcMin = 0;                      // XTC minimum
    public boolean xtcProbabilisticThreshold = false; // Probabilistic threshold

    // Dynamic temperature
    public float dynTempRange = 0.0f;           // Dynamic temp range
    public float dynTempExponent = 1.0f;        // Dynamic temp exponent

    // Guidance
    public float guidanceScale = 1.0f;          // CFG guidance scale
    public String negativePrompt = "";          // Negative prompt for guidance

    // Speculative decoding
    public int nDraft = -1;                     // Number of draft tokens
    public float pAccept = 0.0f;                // Acceptance probability
    public float pSplit = 0.0f;                 // Split probability

    // Output control
    // stream parameter removed - streaming is controlled by tokenCallback presence
    public boolean ignoreEos = false;           // Ignore end-of-sequence token
    public boolean noDisplayPrompt = false;     // Don't display prompt
    public boolean noParseSpecialTokens = false;// Don't parse special tokens

    // Conversation
    public boolean conversation = false;        // Enable conversation mode
    public boolean chatTemplate = true;         // Use chat template
    public boolean escape = true;               // Process escape sequences
    public boolean multiline = false;           // Enable multiline input

    // Token processing
    public boolean addBos = false;              // Add beginning-of-sequence
    public boolean noPenalizeEos = false;       // Don't penalize EOS
    public boolean truncatePrompt = false;      // Truncate prompt if too long

    // Special tokens
    public String bosToken = "";                // Beginning-of-sequence token
    public String eosToken = "";                // End-of-sequence token

    // Infill mode
    public boolean infill = false;              // Enable infill mode
    public String inputPrefix = "";             // Infill prefix
    public String inputSuffix = "";             // Infill suffix

    // Cache control
    public boolean cachePrompt = true;          // Cache the prompt
    public String cacheSession = "";            // Session file for cache

    // Token callback for streaming
    public Predicate<String> tokenCallback;     // Return false to stop generation

    // Logging
    public boolean verbose = false;             // Verbose output
    public boolean displayPrompt = true;        // Display the prompt

    // Legacy parameters for backward compatibility
    public boolean useConversation = false;
    public boolean addAssistant = true;
    public int nThreads = 4;
    public int nBatch = 512;
}
