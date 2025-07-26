import org.llm.wrapper.*;
import java.util.Scanner;

/**
 * Пример тестирования кэширования больших контекстов
 * Демонстрирует эффективность кэширования для больших документов с маленькими вопросами
 */
public class LargePrefixCacheExample {
    public static void main(String[] args) {
        // Проверяем аргументы
        if (args.length < 1) {
            System.err.println("Usage: java LargePrefixCacheExample <model_path>");
            System.exit(1);
        }

        // Инициализация параметров
        InitParams initParams = new InitParams();
        initParams.modelPath = args[0];
        initParams.nCtx = 8192;  // Очень большой контекст для работы с документами
        initParams.nBatch = 512;
        initParams.nThreads = Runtime.getRuntime().availableProcessors();
        initParams.nGpuLayers = 35;  // Загрузить слои на GPU если доступно
        initParams.seed = 42;  // Фиксированный seed для воспроизводимости

        System.out.println("Инициализация модели...");
        long handle = LlamaCpp.llama_init(initParams);
        if (handle == 0) {
            System.err.println("Ошибка инициализации: " + LlamaCpp.llama_get_error());
            System.exit(1);
        }

        try {
            // Очень большой базовый документ (статья о машинном обучении)
            String baseDocument = """
                Машинное обучение — это область искусственного интеллекта, которая изучает
                алгоритмы и модели, позволяющие компьютерам обучаться на данных и делать
                прогнозы или принимать решения без явного программирования для каждой
                конкретной задачи.

                Основные типы машинного обучения:
                1. Обучение с учителем (Supervised Learning)
                2. Обучение без учителя (Unsupervised Learning)
                3. Обучение с подкреплением (Reinforcement Learning)

                Обучение с учителем — это тип машинного обучения, при котором алгоритм обучается
                на размеченных данных. Размеченные данные содержат входные признаки и соответствующие
                им целевые значения. Цель алгоритма — научиться отображать входные данные на
                правильные выходные значения. Примеры алгоритмов обучения с учителем включают
                линейную регрессию, логистическую регрессию, деревья решений, случайные леса,
                градиентный бустинг и нейронные сети.

                Обучение без учителя — это тип машинного обучения, при котором алгоритм обучается
                на неразмеченных данных. Цель алгоритма — найти скрытые структуры или шаблоны в данных.
                Примеры алгоритмов обучения без учителя включают кластеризацию (например, K-means),
                уменьшение размерности (например, PCA, t-SNE) и обнаружение аномалий.

                Обучение с подкреплением — это тип машинного обучения, при котором агент учится
                взаимодействовать с окружающей средой, получая вознаграждения или штрафы за свои
                действия. Цель агента — максимизировать общее вознаграждение. Примеры алгоритмов
                обучения с подкреплением включают Q-learning, SARSA, DQN и PPO.

                Нейронные сети являются одним из ключевых инструментов машинного обучения.
                Они состоят из слоев нейронов, которые обрабатывают информацию и
                передают сигналы друг другу. Каждый нейрон получает входные сигналы,
                применяет к ним функцию активации и передает результат следующему слою.

                Глубокие нейронные сети (Deep Neural Networks) содержат множество скрытых слоев
                между входным и выходным слоями. Это позволяет им изучать сложные иерархические
                представления данных. Глубокие нейронные сети успешно применяются в таких областях,
                как компьютерное зрение, обработка естественного языка и распознавание речи.

                Сверточные нейронные сети (Convolutional Neural Networks, CNN) специализируются
                на обработке данных с сеточной структурой, таких как изображения. Они используют
                операцию свертки для извлечения признаков из входных данных. CNN состоят из
                сверточных слоев, слоев подвыборки (pooling) и полносвязных слоев.

                Рекуррентные нейронные сети (Recurrent Neural Networks, RNN) предназначены для
                обработки последовательных данных, таких как текст или временные ряды. Они имеют
                обратные связи, позволяющие информации персистировать. LSTM (Long Short-Term Memory)
                и GRU (Gated Recurrent Unit) — это специальные типы RNN, которые решают проблему
                исчезающего градиента.

                Трансформеры (Transformers) — это архитектура нейронных сетей, основанная на
                механизме внимания (attention mechanism). Они параллельно обрабатывают все элементы
                последовательности, что делает их более эффективными для обработки длинных
                последовательностей по сравнению с RNN. Трансформеры лежат в основе таких моделей,
                как BERT, GPT и T5.

                Генеративно-состязательные сети (Generative Adversarial Networks, GAN) состоят из
                двух нейронных сетей: генератора и дискриминатора. Генератор создает синтетические
                данные, а дискриминатор пытается отличить их от реальных данных. Они обучаются в
                состязательном процессе, что позволяет генератору создавать все более реалистичные
                данные.

                Автокодировщики (Autoencoders) — это нейронные сети, которые обучаются копировать
                свои входные данные на выход. Они состоят из кодировщика, который сжимает входные
                данные в скрытое представление, и декодировщика, который восстанавливает входные
                данные из этого представления. Автокодировщики используются для уменьшения
                размерности, шумоподавления и генерации данных.
                """;

            // Массив коротких вопросов к документу
            String[] questions = {
                "Что такое машинное обучение?",
                "Какие существуют типы машинного обучения?",
                "Что такое нейронные сети?",
                "Что такое CNN?",
                "Что такое RNN?",
                "Что такое трансформеры?",
                "Что такое GAN?",
                "Что такое автокодировщики?"
            };

            Scanner scanner = new Scanner(System.in);

            // Измеряем размер документа
            System.out.println("\n=== Информация о тесте ===");
            System.out.println("Размер базового документа: " + baseDocument.length() + " символов");
            System.out.println("Количество вопросов: " + questions.length);
            System.out.println("Размер контекста модели: " + initParams.nCtx + " токенов");
            System.out.println("\nНажмите Enter для начала теста...");
            scanner.nextLine();

            // Первый проход - без кэширования (очищаем кэш перед каждым вопросом)
            System.out.println("\n=== Тест 1: Без кэширования (очистка кэша перед каждым вопросом) ===");
            long totalTimeWithoutCache = 0;

            for (int i = 0; i < questions.length; i++) {
                // Очищаем кэш перед каждым вопросом
                LlamaCpp.llama_clear_chat(handle);
                
                String prompt = baseDocument + "\nВопрос: " + questions[i] + "\nОтвет:";

                System.out.println("\n--- Вопрос " + (i + 1) + ": " + questions[i] + " ---");

                long startTime = System.currentTimeMillis();

                // Параметры генерации
                GenerateParams genParams = new GenerateParams();
                genParams.prompt = prompt;
                genParams.nPredict = 100;  // Короткий ответ для быстроты теста
                genParams.temp = 0.7f;
                genParams.topK = 40;
                genParams.topP = 0.95f;
                genParams.repeatPenalty = 1.1f;

                // Callback для вывода токенов по мере генерации
                StringBuilder responseBuilder = new StringBuilder();
                genParams.tokenCallback = token -> {
                    System.out.print(token);
                    System.out.flush();
                    responseBuilder.append(token);
                    return true;  // Продолжить генерацию
                };

                int result = LlamaCpp.llama_generate(handle, genParams);

                long endTime = System.currentTimeMillis();
                long elapsed = endTime - startTime;
                totalTimeWithoutCache += elapsed;

                if (result != 0) {
                    System.err.println("\nОшибка генерации: " + LlamaCpp.llama_get_error());
                    break;
                }

                System.out.println("\nВремя генерации: " + elapsed + " мс");
                System.out.println("Длина ответа: " + responseBuilder.length() + " символов");

                if (i < questions.length - 1) {
                    System.out.println("\nНажмите Enter для следующего вопроса...");
                    scanner.nextLine();
                }
            }

            System.out.println("\n=== Результаты теста без кэширования ===");
            System.out.println("Общее время: " + totalTimeWithoutCache + " мс");
            System.out.println("Среднее время на вопрос: " + (totalTimeWithoutCache / questions.length) + " мс");

            System.out.println("\nНажмите Enter для начала теста с кэшированием...");
            scanner.nextLine();

            // Второй проход - с кэшированием (без очистки кэша между вопросами)
            System.out.println("\n=== Тест 2: С кэшированием (переиспользование кэша между вопросами) ===");
            
            // Очищаем кэш перед началом теста
            LlamaCpp.llama_clear_chat(handle);
            
            long totalTimeWithCache = 0;

            for (int i = 0; i < questions.length; i++) {
                String prompt = baseDocument + "\nВопрос: " + questions[i] + "\nОтвет:";

                System.out.println("\n--- Вопрос " + (i + 1) + ": " + questions[i] + " ---");

                long startTime = System.currentTimeMillis();

                // Параметры генерации
                GenerateParams genParams = new GenerateParams();
                genParams.prompt = prompt;
                genParams.nPredict = 100;  // Короткий ответ для быстроты теста
                genParams.temp = 0.7f;
                genParams.topK = 40;
                genParams.topP = 0.95f;
                genParams.repeatPenalty = 1.1f;

                // Callback для вывода токенов по мере генерации
                StringBuilder responseBuilder = new StringBuilder();
                genParams.tokenCallback = token -> {
                    System.out.print(token);
                    System.out.flush();
                    responseBuilder.append(token);
                    return true;  // Продолжить генерацию
                };

                int result = LlamaCpp.llama_generate(handle, genParams);

                long endTime = System.currentTimeMillis();
                long elapsed = endTime - startTime;
                totalTimeWithCache += elapsed;

                if (result != 0) {
                    System.err.println("\nОшибка генерации: " + LlamaCpp.llama_get_error());
                    break;
                }

                System.out.println("\nВремя генерации: " + elapsed + " мс");
                System.out.println("Длина ответа: " + responseBuilder.length() + " символов");

                // Показываем использование памяти
                long memoryUsage = LlamaCpp.llama_get_native_memory_usage(handle);
                long osMemory = LlamaCpp.llama_get_os_memory_usage();
                System.out.println("Использование памяти: " + (memoryUsage / 1024 / 1024) + " MB");

                if (i < questions.length - 1) {
                    System.out.println("\nНажмите Enter для следующего вопроса...");
                    scanner.nextLine();
                }
            }

            System.out.println("\n=== Результаты теста с кэшированием ===");
            System.out.println("Общее время: " + totalTimeWithCache + " мс");
            System.out.println("Среднее время на вопрос: " + (totalTimeWithCache / questions.length) + " мс");

            // Сравнение результатов
            System.out.println("\n=== Сравнение результатов ===");
            System.out.println("Время без кэширования: " + totalTimeWithoutCache + " мс");
            System.out.println("Время с кэшированием: " + totalTimeWithCache + " мс");
            
            double speedup = (double) totalTimeWithoutCache / totalTimeWithCache;
            System.out.println("Ускорение: " + String.format("%.2f", speedup) + "x");
            System.out.println("Экономия времени: " + String.format("%.2f", (1 - 1/speedup) * 100) + "%");

        } finally {
            // Освобождаем ресурсы
            System.out.println("\n\nОсвобождение ресурсов...");
            LlamaCpp.llama_destroy(handle);
        }

        System.out.println("Программа завершена.");
    }
}
