import csv
import random

def generate_discriminant_dataset(filename="discriminant_data.csv", samples=1000, a_range=(0.1, 10.0), b_range=(0.1, 10.0), c_range=(0.1, 10.0)):
    """
    Генерирует CSV-файл с данными для обучения нейросети.
    Формат: a, b, c, D, где D = b^2 - 4ac
    Диапазон значений можно менять.
    """

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(["a", "b", "c", "discriminant"])

        for _ in range(samples):
            a = random.uniform(*a_range)
            b = random.uniform(*b_range)
            c = random.uniform(*c_range)
            D = b ** 2 - 4 * a * c  # классическая формула дискриминанта

            writer.writerow([a, b, c, D])

    print(f"✅ Файл '{filename}' успешно создан ({samples} строк).")
    print(f"Диапазоны: a={a_range}, b={b_range}, c={c_range}")
    print("Пример формулы: D = b² - 4ac")


if __name__ == "__main__":
    # можно менять диапазоны (для обучения лучше 0–10 или 0–100)
    generate_discriminant_dataset(
        filename="discriminant_data.csv",
        samples=1000,
        a_range=(0.1, 100.0),
        b_range=(0.1, 100.0),
        c_range=(0.1, 100.0)
    )
