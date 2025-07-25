import org.llm.wrapper.*;

public class SimpleTest {
    public static void main(String[] args) {
        if (args.length < 2) {
            System.err.println("Usage: java SimpleTest <model_path> <prompt>");
            System.exit(1);
        }

        LlamaCpp llama = new LlamaCpp();

        // Инициализация
        InitParams init = new InitParams();
        init.modelPath = args[0];
        init.nCtx = 2048;

        long handle = llama.llama_init(init);
        if (handle == 0) {
            System.err.println("Failed to load model");
            return;
        }

        // Генерация
        GenerateParams gen = new GenerateParams();
        gen.prompt = args[1];
        gen.nPredict = 100;
        gen.tokenCallback = token -> {
            System.out.print(token);
            return true;
        };

        llama.llama_generate(handle, gen);

        // Очистка
        llama.llama_destroy(handle);
    }
}
