using Microsoft.ML;
using console_machine_learning.Models;

namespace console_machine_learning.Services;

public class SentimentAnalysisService
{
    private readonly MLContext _mlContext = new();
    private PredictionEngine<SentimentData, SentimentPrediction> _predictionEngine = null!;
    private readonly string _modelPath = "sentiment_model.zip";

    public SentimentAnalysisService()
    {
        if (File.Exists(_modelPath))
            LoadModel();
        else
            TrainAndSaveModel();
    }

    private void TrainAndSaveModel()
    {
        var trainingData = new[]
        {
            new SentimentData { Text = "Eu adorei este produto", Label = true },
            new SentimentData { Text = "O atendimento foi excelente", Label = true },
            new SentimentData { Text = "Isso foi terrível", Label = false },
            new SentimentData { Text = "Estou muito decepcionado", Label = false },
            new SentimentData { Text = "Muito bom, recomendo!", Label = true },
            new SentimentData { Text = "O produto é horrível", Label = false },
            new SentimentData { Text = "Simplesmente perfeito", Label = true },
            new SentimentData { Text = "Foi maravilhoso!", Label = true },
            new SentimentData { Text = "Gostei muito", Label = true },
            new SentimentData { Text = "Funcionou perfeitamente", Label = true },
            new SentimentData { Text = "Produto sensacional", Label = true },
            new SentimentData { Text = "Incrível", Label = true },
            new SentimentData { Text = "Isso foi terrível", Label = false },
            new SentimentData { Text = "Não gostei de nada", Label = false },
            new SentimentData { Text = "Péssimo atendimento", Label = false },
            new SentimentData { Text = "Uma decepção total", Label = false },
            new SentimentData { Text = "Horrível", Label = false },
            new SentimentData { Text = "Não recomendo", Label = false },
            new SentimentData { Text = "Lixo", Label = false }
        };

        var trainData = _mlContext.Data.LoadFromEnumerable(trainingData);

        var pipeline = _mlContext.Transforms.Text.FeaturizeText("Features", nameof(SentimentData.Text))
            .Append(_mlContext.BinaryClassification.Trainers.SdcaLogisticRegression());

        var model = pipeline.Fit(trainData);

        _predictionEngine = _mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

        SaveModel(model, trainData.Schema);
    }

    private void SaveModel(ITransformer model, DataViewSchema schema) =>
        _mlContext.Model.Save(model, schema, _modelPath);

    private void LoadModel()
    {
        using var fileStream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read);
        var model = _mlContext.Model.Load(fileStream, out var schema);

        _predictionEngine = _mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
    }

    public SentimentPrediction Predict(string text)
    {
        var input = new SentimentData { Text = text };

        return _predictionEngine.Predict(input);
    }
}
