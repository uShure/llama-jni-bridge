import org.llm.wrapper.*;

/**
 * Пример использования параметра n_keep для управления кэшем
 */
public class NKeepExample {
    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: java NKeepExample <model_path>");
            System.exit(1);
        }

        // Тестируем разные значения n_keep
        testNKeep(args[0], 0);    // Без кэширования
        testNKeep(args[0], 10);   // Кэшировать только первые 10 токенов
        testNKeep(args[0], 50);   // Кэшировать только первые 50 токенов
        testNKeep(args[0], -1);   // Кэшировать все токены
    }

    private static void testNKeep(String modelPath, int nKeep) {
        System.out.println("\n\n=== Тест с n_keep = " + nKeep + " ===");

        InitParams initParams = new InitParams();
        initParams.modelPath = modelPath;
        initParams.nCtx = 2048;
        initParams.nBatch = 512;
        initParams.nThreads = 4;
        initParams.nKeep = nKeep;  // Устанавливаем n_keep
        initParams.seed = 42;

        long handle = LlamaCpp.llama_init(initParams);
        if (handle == 0) {
            System.err.println("Ошибка инициализации: " + LlamaCpp.llama_get_error());
            return;
        }

        try {
            // Базовый контекст (система инструкций)
            String systemPrompt = "Ты полезный ассистент, который отвечает кратко и по существу.";

            // Первый вопрос
            String prompt1 = systemPrompt + "\nПользователь: Что такое Java?\nАссистент:";
            generateAndMeasure(handle, prompt1, "Вопрос 1");

            // Второй вопрос с тем же системным промптом
            String prompt2 = systemPrompt + "\nПользователь: Что такое Python?\nАссистент:";
            generateAndMeasure(handle, prompt2, "Вопрос 2");

            // Третий вопрос с тем же системным промптом
            String prompt3 = systemPrompt + "\nПользователь: Что такое C++?\nАссистент:";
            generateAndMeasure(handle, prompt3, "Вопрос 3");

            // Очистка чата
            System.out.println("\nОчистка чата...");
            LlamaCpp.llama_clear_chat(handle);

            // Вопрос после очистки
            String prompt4 = systemPrompt + "\nПользователь: Что такое Rust?\nАссистент:";
            generateAndMeasure(handle, prompt4, "Вопрос после очистки");

        } finally {
            LlamaCpp.llama_destroy(handle);
        }
    }

    private static void generateAndMeasure(long handle, String prompt, String label) {
        System.out.println("\n--- " + label + " ---");
        System.out.println("Промпт: " + prompt.substring(Math.max(0, prompt.length() - 50)) + "...");

        long startTime = System.currentTimeMillis();

        GenerateParams genParams = new GenerateParams();
        genParams.prompt = prompt;
        genParams.nPredict = 50;  // Короткие ответы для быстрого теста
        genParams.temp = 0.1f;    // Низкая температура для стабильности
        //genParams.stream = true;

        final StringBuilder response = new StringBuilder();
        genParams.tokenCallback = token -> {
            response.append(token);
            System.out.print(token);
            System.out.flush();
            return true;
        };

        int result = LlamaCpp.llama_generate(handle, genParams);

        long endTime = System.currentTimeMillis();

        if (result != 0) {
            System.err.println("\nОшибка генерации: " + LlamaCpp.llama_get_error());
        } else {
            System.out.println("\n\nВремя генерации: " + (endTime - startTime) + " мс");
            System.out.println("Длина ответа: " + response.length() + " символов");
        }
    }
}
