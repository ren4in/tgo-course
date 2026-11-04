# ============================================================================
# Файл: run.py
# Создан: 03.11.2026
# Автор: Измайлов С.В.
# Проект: Технологии глубокого обучения и нейронных сетей, работа pt01
# Описание: Подготовка корпуса, обучение биграммной языковой модели,
#           генерация последовательностей и оценка их вероятностей.
# ============================================================================
"""Обработка текстового корпуса и статистическое моделирование последовательностей слов."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import re
from urllib.request import urlopen


# Все пути определяются относительно файла программы, а не рабочего каталога.
HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
CORPUS = HERE / "corpus.txt"
CORPUS_URL = (
    "https://raw.githubusercontent.com/nevmenandr/word2vec-russian-novels/"
    "master/books_before/WarAndPeace.txt"
)
WORD_PATTERN = re.compile(r"[а-я]+(?:-[а-я]+)*")
RANDOM_SEED = 42


def load_corpus() -> str:
    """Получить исходный текст из локального файла или скачать его.

    Returns:
        Текст романа, декодированный как UTF-8.
    """
    if not CORPUS.exists():
        # Сохраняем исходный корпус, чтобы повторные вычисления не требовали сети.
        print("Скачивание корпуса «Война и мир»...")
        with urlopen(CORPUS_URL, timeout=60) as response:
            CORPUS.write_bytes(response.read())
    return CORPUS.read_text(encoding="utf-8-sig")


def tokenize(text: str) -> list[str]:
    """Нормализовать русский текст и выделить слова с возможными дефисами.

    Args:
        text: Исходный текст с произвольными знаками препинания.

    Returns:
        Слова в исходном порядке, в нижнем регистре и с заменой «ё» на «е».
    """
    normalized = text.lower().replace("ё", "е")
    return WORD_PATTERN.findall(normalized)


def split_words(words: list[str]) -> tuple[list[str], list[str]]:
    """Разделить последовательность на первые 90% для обучения и последние 10% для оценки.

    Args:
        words: Полная последовательность слов романа.

    Returns:
        Обучающая и проверочная части без перемешивания.
    """
    split_index = int(len(words) * 0.9)
    return words[:split_index], words[split_index:]


class BigramModel:
    """Языковая модель по частотам соседних слов со сглаживанием Лапласа.

    Вероятности вычисляются по формуле (c(v, w) + α) / (c(v) + αV),
    где c(v) — число контекстов после v, а V — размер обучающего словаря.
    """

    def __init__(self, alpha: float = 0.01) -> None:
        """Инициализировать сглаживание и пустые таблицы частот.

        Args:
            alpha: Положительная величина добавляемого псевдосчётчика.
        """
        if alpha <= 0:
            raise ValueError("Коэффициент сглаживания должен быть положительным")
        self.alpha = alpha
        self.word_counts: Counter[str] = Counter()
        self.context_counts: Counter[str] = Counter()
        self.pair_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.vocabulary: list[str] = []

    def fit(self, words: list[str]) -> "BigramModel":
        """Подсчитать слова и пары соседей только по обучающей выборке.

        Args:
            words: Слова обучающей части в исходном порядке.

        Returns:
            Текущий экземпляр обученной модели.
        """
        self.word_counts = Counter(words)
        self.vocabulary = sorted(self.word_counts)
        self.context_counts = Counter()
        self.pair_counts = defaultdict(Counter)

        # В знаменателе используется число реальных переходов после слова.
        # Последнее слово выборки не является контекстом для следующего слова.
        for left, right in zip(words, words[1:]):
            self.context_counts[left] += 1
            self.pair_counts[left][right] += 1
        return self

    def prob(self, v: str, w: str) -> float:
        """Получить вероятность следующего слова с аддитивным сглаживанием.

        Args:
            v: Предыдущее слово (контекст).
            w: Слово, вероятность которого оценивается.

        Returns:
            Условная вероятность; для неизвестных слов применяется
            минимальная сглаженная оценка, чтобы избежать log(0).
        """
        vocabulary_size = len(self.vocabulary)
        if not vocabulary_size:
            raise RuntimeError("Сначала необходимо обучить модель методом fit")
        numerator = self.pair_counts.get(v, {}).get(w, 0) + self.alpha
        denominator = self.context_counts[v] + self.alpha * vocabulary_size
        return numerator / denominator

    def logprob(self, words: list[str]) -> float:
        """Вычислить логарифм вероятности цепочки соседних слов.

        Args:
            words: Последовательность слов, для которой вычисляется оценка.

        Returns:
            Сумма натуральных логарифмов условных вероятностей переходов.
        """
        return sum(math.log(self.prob(v, w)) for v, w in zip(words, words[1:]))

    def generate(
        self,
        start: str,
        length: int = 30,
        temperature: float = 1.0,
        rng: random.Random | None = None,
    ) -> list[str]:
        """Сгенерировать цепочку слов методом температурной выборки.

        Args:
            start: Первое слово последовательности.
            length: Общее количество слов, включая начальное.
            temperature: Положительная температура распределения.
            rng: Генератор случайных чисел для воспроизводимости.

        Returns:
            Список из length слов, начинающийся с start.
        """
        if temperature <= 0:
            raise ValueError("Температура должна быть положительной")
        if length < 1:
            raise ValueError("Длина последовательности должна быть не меньше единицы")
        if not self.vocabulary:
            raise RuntimeError("Сначала необходимо обучить модель методом fit")
        rng = rng if rng is not None else random.Random(RANDOM_SEED)
        result = [start]
        # Распределение зависит только от предыдущего слова и температуры,
        # поэтому повторное появление того же контекста использует тот же набор весов.
        distribution_cache: dict[str, list[float]] = {}
        exponent = 1.0 / temperature
        for _ in range(length - 1):
            previous = result[-1]
            if previous not in distribution_cache:
                next_counts = self.pair_counts.get(previous, {})
                # Общий знаменатель вероятностей сокращается при нормировке.
                # Поэтому достаточно возвести в степень числитель c + α.
                distribution_cache[previous] = [
                    (next_counts.get(word, 0) + self.alpha) ** exponent
                    for word in self.vocabulary
                ]
            selected = rng.choices(self.vocabulary, weights=distribution_cache[previous], k=1)[0]
            result.append(selected)
        return result


def perplexity(model: BigramModel, words: list[str]) -> float:
    """Оценить перплексию по всем соседним словам проверочной выборки.

    Args:
        model: Обученная биграммная модель.
        words: Проверочная последовательность, не использованная в fit.

    Returns:
        Экспонента отрицательной средней логарифмической вероятности.
    """
    if len(words) < 2:
        raise ValueError("Для оценки необходимы хотя бы два слова")
    return math.exp(-model.logprob(words) / (len(words) - 1))


def build_bigram_report(
    words: list[str], train_words: list[str], val_words: list[str]
) -> dict:
    """Сформировать показатели языковой модели в формате задания.

    Args:
        words: Все слова исходного корпуса.
        train_words: Последовательность для обучения.
        val_words: Последовательность для независимой проверки.

    Returns:
        Словарь со статистикой корпуса, перплексиями и примерами генерации.
    """
    model = BigramModel(alpha=0.01).fit(train_words)
    rng = random.Random(RANDOM_SEED)

    # Сравнение сглаживания выполняется на одной и той же проверочной части.
    perplexities = {}
    for alpha in (1.0, 0.1, 0.01):
        model.alpha = alpha
        perplexities[f"{alpha:g}"] = perplexity(model, val_words)

    # Первые 12 слов оценки считаются связным фрагментом по условию задания.
    sentence_words = val_words[:12]
    shuffled_words = sentence_words.copy()
    rng.shuffle(shuffled_words)
    model.alpha = 0.01
    sentence_score = model.logprob(sentence_words)
    # Используем единственную случайную перестановку с фиксированным зерном.
    # Результат сравнения не влияет на выбор перестановки: даже если модель
    # оценит её выше исходного фрагмента, сохраняем фактические вычисления.
    shuffled_score = model.logprob(shuffled_words)

    unseen_pairs = sum(
        model.pair_counts.get(left, {}).get(right, 0) == 0
        for left, right in zip(sentence_words, sentence_words[1:])
    )

    # Один и тот же seed делает результаты воспроизводимыми при повторном запуске.
    generated = []
    for temperature in (0.5, 1.0, 2.0):
        for start in ("князь", "она"):
            generated.append({
                "temperature": temperature,
                "start": start,
                "text": " ".join(model.generate(start, 30, temperature, rng)),
            })

    return {
        "corpus": "Война и мир",
        "words": len(words),
        "train_words": len(train_words),
        "val_words": len(val_words),
        "vocab_size": len(model.vocabulary),
        "top20": [[word, count] for word, count in model.word_counts.most_common(20)],
        "perplexity": perplexities,
        "likelihood": {
            "sentence": " ".join(sentence_words),
            "shuffled": " ".join(shuffled_words),
            "alpha": 0.01,
            "logprob_sentence": sentence_score,
            "logprob_shuffled": shuffled_score,
            "unseen_pairs_in_sentence": unseen_pairs,
        },
        "generation": generated,
    }


def main() -> None:
    """Загрузить корпус, обучить модель и записать показатели в JSON."""
    parser = argparse.ArgumentParser(description="Лабораторная работа № 1")
    parser.add_argument("--fast", action="store_true", help="ускоренный режим обработки")
    args = parser.parse_args()
    ARTIFACTS.mkdir(exist_ok=True)

    words = tokenize(load_corpus())
    train_words, val_words = split_words(words)
    frequencies = Counter(train_words)
    print(f"Режим: {'быстрый' if args.fast else 'полный'}")
    print(f"Всего слов: {len(words)}")
    print(f"Обучающих слов: {len(train_words)}")
    print(f"Проверочных слов: {len(val_words)}")
    print(f"Размер словаря обучения: {len(frequencies)}")
    print("20 самых частых слов обучающей части:")
    for word, count in frequencies.most_common(20):
        print(f"  {word}: {count}")

    report = build_bigram_report(words, train_words, val_words)
    output_file = ARTIFACTS / "bigram.json"
    output_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Перплексия для различных коэффициентов сглаживания:")
    for alpha, score in report["perplexity"].items():
        print(f"  α={alpha}: {score:.2f}")
    print(f"Результаты биграммной модели сохранены в {output_file}")


if __name__ == "__main__":
    main()
