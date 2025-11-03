import math
import csv
import numpy as np

# ============================================================
# Класс нейросети
# ============================================================
class NeuralNetwork:
    def __init__(self, hiddenWeights, outputWeights, hiddenBias, outputBias, requestedEra=1000, lr=0.01, momentum=0.1):
        self.requestedEra = requestedEra
        self.hiddenWeights = hiddenWeights
        self.outputWeights = outputWeights
        self.hiddenBias = hiddenBias
        self.outputBias = outputBias
        self.lr = lr
        self.momentum = momentum

        self.dataset = []
        self.answersList = []

        # Для хранения дельт
        self.hiddenWeightDeltas = np.zeros_like(hiddenWeights)
        self.outputWeightDeltas = np.zeros_like(outputWeights)
        self.hiddenBiasDeltas = np.zeros_like(hiddenBias)
        self.outputBiasDelta = 0.0

    # ------------------------------------------------------------
    # Чтение данных
    # ------------------------------------------------------------
    def readFromCsv(self, datasetPath: str):
        with open(datasetPath, 'r') as csvDataset:
            reader = csv.reader(csvDataset, delimiter='\t')
            next(reader)  # пропускаем заголовок
            for row in reader:
                a, b, c, d = map(float, row)
                self.dataset.append([a, b, c, d])
        self._normalizeDataset()
        print("Нормализация входов выполнена.")
    #новая функция нормализации
    def _normalizeDataset(self):
        data = np.array(self.dataset)
        inputs = data[:, :3]
        outputs = data[:, 3]

        # Z-score нормализация входов
        self.input_means = inputs.mean(axis=0)
        self.input_stds = inputs.std(axis=0)
        self.input_stds[self.input_stds == 0] = 1e-6
        self.normalized_inputs = (inputs - self.input_means) / self.input_stds

        # Масштабирование выхода в [-1, 1] для стабильности
        self.output_scale = np.max(np.abs(outputs))
        self.normalized_outputs = outputs / self.output_scale

        self.normalized_dataset = np.hstack((self.normalized_inputs, self.normalized_outputs.reshape(-1, 1)))

    # новая функция денормализации
    def denormalize_output(self, y):
        return y * self.output_scale

    # ------------------------------------------------------------
    # Прямое распространение
    # ------------------------------------------------------------
    #Биасы добавлены + выходной нейрон изменил
    def forward(self, x):
        hidden_input = np.dot(self.hiddenWeights, x) + self.hiddenBias
        hidden_output = np.tanh(hidden_input)
        output_input = np.dot(self.outputWeights, hidden_output) + self.outputBias
        output = output_input  # линейный выход
        return hidden_output, output

    # ------------------------------------------------------------
    # Обратное распространение
    # ------------------------------------------------------------
    def backward(self, x, hidden_output, output, target):
        error = target - output
        d_output = error  # производная линейной функции = 1

        d_hidden = (1 - hidden_output ** 2) * (self.outputWeights * d_output)

        # обновляем веса с моментом
        output_deltas = self.lr * d_output * hidden_output + self.momentum * self.outputWeightDeltas
        hidden_deltas = self.lr * np.outer(d_hidden, x) + self.momentum * self.hiddenWeightDeltas

        # обновляем смещения
        output_bias_delta = self.lr * d_output + self.momentum * self.outputBiasDelta
        hidden_bias_deltas = self.lr * d_hidden + self.momentum * self.hiddenBiasDeltas

        # применяем обновления
        self.outputWeights += output_deltas
        self.hiddenWeights += hidden_deltas
        self.outputBias += output_bias_delta
        self.hiddenBias += hidden_bias_deltas

        # сохраняем дельты для момента
        self.outputWeightDeltas = output_deltas
        self.hiddenWeightDeltas = hidden_deltas
        self.outputBiasDelta = output_bias_delta
        self.hiddenBiasDeltas = hidden_bias_deltas

        return error ** 2

    # ------------------------------------------------------------
    # Обучение
    # ------------------------------------------------------------
    #Mse считается на нормализованных выходах
    def train(self):
        for era in range(self.requestedEra):
            mse = 0.0
            np.random.shuffle(self.normalized_dataset)
            for sample in self.normalized_dataset:
                x = sample[:3]
                target = sample[3]
                hidden_output, output = self.forward(x)
                mse += self.backward(x, hidden_output, output, target)
            #Адаптивная скорость обучения у себя не делал хоть у тебя потыаюсь
            mse /= len(self.normalized_dataset)
            if era % 100 == 0:
                print(f"Эпоха {era:5d}, MSE: {mse:.6f}, LR: {self.lr:.4f}")
            if mse < 1e-5:
                print(f"Обучение завершено на эпохе {era}")
                break

    # ------------------------------------------------------------
    # Предсказание
    # ------------------------------------------------------------
    #Эта херня для денормализации(тесты и валидация)
    def predict(self, a, b, c):
        x = np.array([a, b, c])
        x_norm = (x - self.input_means) / self.input_stds
        _, y_norm = self.forward(x_norm)
        return self.denormalize_output(y_norm)


# ============================================================
# Запуск
# ============================================================

if __name__ == "__main__":
    hiddenWeights = np.array([
        [0.1, 0.9, 0.3],
        [0.4, 0.7, 0.1],
        [0.3, 0.2, 0.2]
    ])
    outputWeights = np.array([0.5, 0.4, 0.6])
    hiddenBias = np.array([0.1, 0.1, 0.1])
    outputBias = 0.1

    print("Начальные веса:")
    print("Hidden:", hiddenWeights)
    print("Output:", outputWeights)
    print("Bias H:", hiddenBias, "Bias O:", outputBias)

    myAI = NeuralNetwork(hiddenWeights, outputWeights, hiddenBias, outputBias, requestedEra=2000)
    myAI.readFromCsv("discriminant_data.csv")
    myAI.train()

    print("\nТестирование:")
    test_cases = [
        (1, 2, 1),
        (1, 5, 1),
        (2, 4, 1),
        (0.5, 3, 2),
        (4, 4, 4)
    ]
    for a, b, c in test_cases:
        pred = myAI.predict(a, b, c)
        actual = b**2 - 4*a*c
        print(f"a={a}, b={b}, c={c} → предсказано: {pred:.3f}, реально: {actual:.3f}, ошибка: {abs(pred-actual):.3f}")
