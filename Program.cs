using console_machine_learning.Services;

var sentimentService = new SentimentAnalysisService();

var inputs = new[]
{
    "Eu amei a experiência",
    "Não gostei de nada",
    "Foi maravilhoso!",
    "Péssimo atendimento",
    "Simplesmente perfeito",
    "O lugar é horrível"
};

foreach (var input in inputs)
{
    var prediction = sentimentService.Predict(input);

    Console.ForegroundColor = ConsoleColor.Magenta;
    Console.Write("Expressão: ");

    Console.ForegroundColor = ConsoleColor.Yellow;
    Console.Write($"'{input}'");

    Console.ForegroundColor = ConsoleColor.Magenta;
    Console.Write(" | Sentimento: ");

    Console.ForegroundColor = ConsoleColor.Yellow;
    Console.Write($"{(prediction.Prediction ? "Bom" : "Ruim")}");

    Console.ForegroundColor = ConsoleColor.Magenta;
    Console.Write(" | Probabilidade: ");

    Console.ForegroundColor = ConsoleColor.Yellow;
    Console.Write($"{prediction.Probability:P2}");

    Console.ForegroundColor = ConsoleColor.Magenta;
    Console.Write(" | Score: ");

    Console.ForegroundColor = ConsoleColor.Yellow;
    Console.WriteLine($"{prediction.Score:F2}");
}
