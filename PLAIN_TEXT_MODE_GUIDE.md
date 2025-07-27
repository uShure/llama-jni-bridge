# Plain Text Mode Guide - Avoiding Hallucinations

## Problem: Forced Chat Templates Causing Hallucinations

When using chat mode, the model may:
- Add unwanted system prompts
- Continue conversations that don't exist
- Generate questions and answer them itself
- Add role markers like "User:" and "Assistant:"

## Solution: Use Plain Text Mode

### Quick Fix
```java
GenerateParams params = new GenerateParams();
params.chatMode = false;  // Disable chat mode
params.prompt = "Your prompt here";
```

## Detailed Examples

### Example 1: Simple Question (Plain Text Mode)
```java
// Plain text mode - clean output
GenerateParams params = new GenerateParams();
params.prompt = "The capital of France is";
params.chatMode = false;  // KEY: Set to false
params.nPredict = 50;
params.temp = 0.7f;

int result = LlamaCpp.llama_generate(handle, params);
// Output: "Paris. It is known for the Eiffel Tower..."
```

### Example 2: Same Question (Chat Mode)
```java
// Chat mode - may add unwanted formatting
GenerateParams params = new GenerateParams();
params.prompt = "The capital of France is";
params.chatMode = true;  // Templates and history enabled
params.nPredict = 50;
params.temp = 0.7f;

int result = LlamaCpp.llama_generate(handle, params);
// Output might be: "User: The capital of France is\nAssistant: The capital of France is Paris.\nUser: What else would you like to know?\nAssistant: ..."
```

## When to Use Each Mode

### Use Plain Text Mode When:
- You want exact control over input/output
- Building your own conversation management
- Implementing custom templates
- Doing completion tasks (not conversation)
- Experiencing hallucinations in chat mode

### Use Chat Mode When:
- You want automatic conversation formatting
- Building a chatbot with history
- The model was specifically trained for chat format
- You need role-based interactions

## Advanced Control

### Custom Conversation Management
```java
public class CustomConversation {
    private List<String> history = new ArrayList<>();
    private long llamaHandle;

    public String generate(String userInput) {
        // Build your own context
        StringBuilder context = new StringBuilder();

        // Add your custom format
        for (String msg : history) {
            context.append(msg).append("\n");
        }
        context.append("Human: ").append(userInput).append("\n");
        context.append("Assistant: ");

        // Use plain text mode
        GenerateParams params = new GenerateParams();
        params.prompt = context.toString();
        params.chatMode = false;  // No templates
        params.nPredict = 200;
        params.stopSequences = new String[]{"Human:", "\n\n"};

        // Generate
        StringBuilder response = new StringBuilder();
        params.tokenCallback = token -> {
            response.append(token);
            return true;
        };

        LlamaCpp.llama_generate(llamaHandle, params);

        // Save to history
        history.add("Human: " + userInput);
        history.add("Assistant: " + response.toString());

        return response.toString();
    }
}
```

### Caching with Plain Text Mode
```java
// First request
params.prompt = "Translate to French: Hello world";
params.chatMode = false;
params.keep = -1;  // Keep all tokens for caching

// Second request - will reuse cache
params.prompt = "Translate to French: Hello world. How are you?";
params.chatMode = false;
// Common prefix "Translate to French: Hello world" is cached
```

## Debugging Hallucinations

If you still see hallucinations in plain text mode:

1. **Check the prompt**: Make sure no chat markers are in your input
2. **Check stop sequences**: Add appropriate stop sequences
3. **Check the model**: Some models are heavily chat-tuned
4. **Monitor output**: Use token callback to stop early if needed

```java
params.tokenCallback = token -> {
    // Stop if model starts hallucinating
    if (token.contains("User:") || token.contains("Assistant:")) {
        return false;  // Stop generation
    }
    return true;
};
```

## Summary

- Set `chatMode = false` for direct control
- Build your own templates if needed
- Use stop sequences to control output
- Monitor generation with callbacks
- Keep caching benefits with proper prompt design

This approach gives you complete control over model behavior and eliminates template-induced hallucinations.
