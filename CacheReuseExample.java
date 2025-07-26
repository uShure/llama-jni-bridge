import org.llm.wrapper.*;
import java.util.Scanner;

/**
 * Пример использования переиспользования кэша для работы с большими документами
 */
public class CacheReuseExample {
    public static void main(String[] args) {
        // Проверяем аргументы
        if (args.length < 1) {
            System.err.println("Usage: java CacheReuseExample <model_path>");
            System.exit(1);
        }

        // Инициализация параметров
        InitParams initParams = new InitParams();
        initParams.modelPath = args[0];
        initParams.nCtx = 4096;  // Большой контекст для работы с документами
        initParams.nBatch = 512;
        initParams.nThreads = Runtime.getRuntime().availableProcessors();
        initParams.nGpuLayers = 35;  // Загрузить слои на GPU если доступно
        initParams.seed = 42;  // Фиксированный seed для воспроизводимости

        // Настройка GPU устройств (если есть несколько GPU)
        // initParams.gpuDevices = new int[]{0, 1};
        // initParams.tensorSplits = new float[]{0.5f, 0.5f};

        System.out.println("Инициализация модели...");
        long handle = LlamaCpp.llama_init(initParams);
        if (handle == 0) {
            System.err.println("Ошибка инициализации: " + LlamaCpp.llama_get_error());
            System.exit(1);
        }

        try {
            // Большой базовый документ
            String baseDocument = """
                Машинное обучение — это область искусственного интеллекта, которая изучает
                алгоритмы и модели, позволяющие компьютерам обучаться на данных и делать
                прогнозы или принимать решения без явного программирования для каждой
                конкретной задачи.

                Основные типы машинного обучения:
                1. Обучение с учителем (Supervised Learning)
                2. Обучение без учителя (Unsupervised Learning)
                3. Обучение с подкреплением (Reinforcement Learning)

                Нейронные сети являются одним из ключевых инструментов машинного обучения.
                Они состоят из слоев нейронов, которые обрабатывают информацию и
                передают сигналы друг другу.

                """;

            // Массив вопросов к документу
            String[] questions = {
                "Что такое машинное обучение?",
                "Какие существуют типы машинного обучения?",
                "Что такое нейронные сети?",
                "Как работают нейронные сети?"
            };

            Scanner scanner = new Scanner(System.in);

            // Обрабатываем каждый вопрос
            for (int i = 0; i < questions.length; i++) {
                String prompt = baseDocument + "\nВопрос: " + questions[i] + "\nОтвет:";

                System.out.println("\n=== Вопрос " + (i + 1) + " ===");
                System.out.println(questions[i]);
                System.out.println("\nГенерация ответа...");

                long startTime = System.currentTimeMillis();

                // Параметры генерации
                GenerateParams genParams = new GenerateParams();
                genParams.prompt = prompt;
                genParams.nPredict = 256;
                genParams.temp = 0.7f;
                genParams.topK = 40;
                genParams.topP = 0.95f;
                genParams.repeatPenalty = 1.1f;
                //genParams.stream = true;

                // Callback для вывода токенов по мере генерации
                genParams.tokenCallback = token -> {
                    System.out.print(token);
                    System.out.flush();
                    return true;  // Продолжить генерацию
                };

                int result = LlamaCpp.llama_generate(handle, genParams);

                long endTime = System.currentTimeMillis();

                if (result != 0) {
                    System.err.println("\nОшибка генерации: " + LlamaCpp.llama_get_error());
                    break;
                }

                System.out.println("\n\nВремя генерации: " + (endTime - startTime) + " мс");

                // Показываем использование памяти
                long memoryUsage = LlamaCpp.llama_get_native_memory_usage(handle);
                long osMemory = LlamaCpp.llama_get_os_memory_usage();
                System.out.println("Использование памяти: " + (memoryUsage / 1024 / 1024) + " MB");
                System.out.println("OS память: " + (osMemory / 1024 / 1024) + " MB");

                if (i < questions.length - 1) {
                    System.out.println("\nНажмите Enter для следующего вопроса...");
                    scanner.nextLine();
                }
            }

            // Демонстрация очистки чата
            System.out.println("\n=== Очистка истории чата ===");
            LlamaCpp.llama_clear_chat(handle);
            System.out.println("История чата и кэш очищены.");

            // Еще один вопрос после очистки
            System.out.println("\nЗадаем вопрос после очистки кэша...");
            GenerateParams genParams = new GenerateParams();
            genParams.prompt = "Что такое искусственный интеллект?";
            genParams.nPredict = 128;
            genParams.stream = true;
            genParams.tokenCallback = token -> {
                System.out.print(token);
                System.out.flush();
                return true;
            };

            LlamaCpp.llama_generate(handle, genParams);

        } finally {
            // Освобождаем ресурсы
            System.out.println("\n\nОсвобождение ресурсов...");
            LlamaCpp.llama_destroy(handle);
        }

        System.out.println("Программа завершена.");
    }
}
