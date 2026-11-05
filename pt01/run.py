# ============================================================================
# Файл: run.py
# Создан: 03.11.2026
# Автор: Измайлов С.В.
# Проект: Технологии глубокого обучения и нейронных сетей, работа pt01
# Описание: Статистическая биграммная модель, оценка текста и собственный BPE.
# ============================================================================
"""Обработка русского текста, биграммная модель и частотная BPE-токенизация."""

import argparse
from collections import Counter, defaultdict
import heapq
import json
import math
from pathlib import Path
import random
import re
from urllib.request import urlopen

# Местоположение артефактов не зависит от рабочего каталога терминала.
HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
CORPUS = HERE / "corpus.txt"
CORPUS_URL = (
    "https://raw.githubusercontent.com/nevmenandr/word2vec-russian-novels/"
    "master/books_before/WarAndPeace.txt"
)
WORD_PATTERN = re.compile(r"[а-я]+(?:-[а-я]+)*")
RANDOM_SEED = 42
END_OF_WORD = "</w>"


def load_corpus() -> str:
    """Прочитать корпус с диска, предварительно скачав его при необходимости.

    Returns:
        Исходный текст романа в кодировке UTF-8.
    """
    if not CORPUS.exists():
        print("Скачивание корпуса «Война и мир»...")
        with urlopen(CORPUS_URL, timeout=60) as response:
            CORPUS.write_bytes(response.read())
    return CORPUS.read_text(encoding="utf-8-sig")


def tokenize(text: str) -> list[str]:
    """Привести текст к нижнему регистру и выделить русские слова.

    Args:
        text: Исходная строка со знаками препинания.

    Returns:
        Список слов, включая сложные слова с дефисом.
    """
    normalized = text.lower().replace("ё", "е")
    return WORD_PATTERN.findall(normalized)


def split_words(words: list[str]) -> tuple[list[str], list[str]]:
    """Выделить первые 90% слов для обучения, остальные — для проверки.

    Args:
        words: Полная последовательность слов.

    Returns:
        Пара списков (обучение, проверка) в исходном порядке.
    """
    border = int(len(words) * 0.9)
    return words[:border], words[border:]


class BigramModel:
    """Оценивать вероятность следующего слова по частотам пар соседей."""

    def __init__(self, alpha: float = 0.01) -> None:
        """Создать модель со сглаживанием Лапласа.

        Args:
            alpha: Положительный коэффициент сглаживания.
        """
        if alpha <= 0:
            raise ValueError("Коэффициент сглаживания должен быть положительным")
        self.alpha = alpha
        self.word_counts: Counter[str] = Counter()
        self.context_counts: Counter[str] = Counter()
        self.pair_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.vocabulary: list[str] = []

    def fit(self, words: list[str]) -> "BigramModel":
        """Обучить модель: подсчитать слова и переходы на обучающих данных.

        Args:
            words: Слова обучающей части в исходном порядке.

        Returns:
            Текущий обученный объект модели.
        """
        self.word_counts = Counter(words)
        self.vocabulary = sorted(self.word_counts)
        self.context_counts = Counter()
        self.pair_counts = defaultdict(Counter)
        # Последнее слово не имеет правого соседа и не входит в контексты.
        for left, right in zip(words, words[1:]):
            self.context_counts[left] += 1
            self.pair_counts[left][right] += 1
        return self

    def prob(self, v: str, w: str) -> float:
        """Вычислить сглаженную вероятность перехода от v к w.

        Args:
            v: Предыдущее слово.
            w: Предполагаемое следующее слово.

        Returns:
            Значение по формуле (c(v,w)+alpha)/(c(v)+alpha*V).
        """
        vocabulary_size = len(self.vocabulary)
        if not vocabulary_size:
            raise RuntimeError("Сначала необходимо обучить модель методом fit")
        numerator = self.pair_counts.get(v, {}).get(w, 0) + self.alpha
        denominator = self.context_counts[v] + self.alpha * vocabulary_size
        return numerator / denominator

    def logprob(self, words: list[str]) -> float:
        """Сложить натуральные логарифмы вероятностей соседних переходов.

        Args:
            words: Оцениваемая последовательность.

        Returns:
            Логарифмическая оценка вероятности последовательности.
        """
        return sum(math.log(self.prob(v, w)) for v, w in zip(words, words[1:]))

    def generate(
        self, start: str, length: int = 30, temperature: float = 1.0,
        rng: random.Random | None = None,
    ) -> list[str]:
        """Создать последовательность случайной выборкой с температурой.

        Args:
            start: Первое слово.
            length: Итоговая длина в словах вместе с первым.
            temperature: Температура изменения распределения вероятностей.
            rng: Источник псевдослучайных чисел.

        Returns:
            Список сгенерированных слов.
        """
        if temperature <= 0:
            raise ValueError("Температура должна быть положительной")
        if length < 1:
            raise ValueError("Длина должна быть положительной")
        if not self.vocabulary:
            raise RuntimeError("Сначала необходимо обучить модель методом fit")
        rng = rng if rng is not None else random.Random(RANDOM_SEED)
        result = [start]
        # Повторно встретившееся предыдущее слово использует готовые веса.
        cache: dict[str, list[float]] = {}
        exponent = 1.0 / temperature
        for _ in range(length - 1):
            previous = result[-1]
            if previous not in cache:
                counts = self.pair_counts.get(previous, {})
                # Знаменатель одинаков для всех кандидатов и сокращается.
                cache[previous] = [
                    (counts.get(word, 0) + self.alpha) ** exponent
                    for word in self.vocabulary
                ]
            result.append(rng.choices(self.vocabulary, weights=cache[previous], k=1)[0])
        return result


def perplexity(model: BigramModel, words: list[str]) -> float:
    """Оценить среднюю неопределённость модели на отложенном корпусе.

    Args:
        model: Обученная языковая модель.
        words: Отложенная последовательность слов.

    Returns:
        Экспонента минус средней логарифмической вероятности.
    """
    if len(words) < 2:
        raise ValueError("Для оценки требуется не меньше двух слов")
    return math.exp(-model.logprob(words) / (len(words) - 1))


def build_bigram_report(
    words: list[str], train_words: list[str], val_words: list[str],
) -> dict:
    """Вычислить статистику биграммной модели для JSON-отчёта.

    Args:
        words: Полный корпус.
        train_words: Обучающая часть.
        val_words: Отложенная часть.

    Returns:
        Словарь показателей, соответствующий формату лабораторной.
    """
    model = BigramModel(alpha=0.01).fit(train_words)
    rng = random.Random(RANDOM_SEED)
    scores = {}
    for alpha in (1.0, 0.1, 0.01):
        model.alpha = alpha
        scores[f"{alpha:g}"] = perplexity(model, val_words)
    sentence = val_words[:12]
    shuffled = sentence.copy()
    # Ровно одна случайная перестановка, без выбора наиболее удобного результата.
    rng.shuffle(shuffled)
    model.alpha = 0.01
    sentence_score = model.logprob(sentence)
    shuffled_score = model.logprob(shuffled)
    unseen = sum(
        model.pair_counts.get(a, {}).get(b, 0) == 0
        for a, b in zip(sentence, sentence[1:])
    )
    samples = []
    for temperature in (0.5, 1.0, 2.0):
        for start in ("князь", "она"):
            samples.append({
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
        "perplexity": scores,
        "likelihood": {
            "sentence": " ".join(sentence),
            "shuffled": " ".join(shuffled),
            "alpha": 0.01,
            "logprob_sentence": sentence_score,
            "logprob_shuffled": shuffled_score,
            "unseen_pairs_in_sentence": unseen,
        },
        "generation": samples,
    }


class BPETokenizer:
    """Частотный BPE-токенизатор с воспроизводимым порядком слияний."""

    def __init__(self) -> None:
        """Подготовить список слияний и их ранги."""
        self.merges: list[tuple[str, str]] = []
        self.ranks: dict[tuple[str, str], int] = {}

    @staticmethod
    def _word_pairs(symbols: tuple[str, ...]) -> set[tuple[str, str]]:
        """Найти уникальные соседние пары внутри одного представления слова.

        Args:
            symbols: Текущее символьное представление слова.

        Returns:
            Множество пар для индекса затронутых слов.
        """
        return set(zip(symbols, symbols[1:]))

    def fit(self, frequencies: Counter[str], num_merges: int, min_freq: int = 2) -> "BPETokenizer":
        """Обучить BPE на частотах слов с инкрементальным пересчётом пар.

        Args:
            frequencies: Частоты уникальных слов полного корпуса.
            num_merges: Желаемое число операций слияния.
            min_freq: Нижняя граница частоты слова для обучения BPE.

        Returns:
            Текущий обученный токенизатор.
        """
        if num_merges < 0 or min_freq < 1:
            raise ValueError("Параметры BPE должны быть неотрицательными")
        self.merges = []
        self.ranks = {}
        # Слова с частотой меньше порога не влияют на выбор слияний.
        # При этом статистику токенов ниже подсчитаем на ВСЁМ корпусе.
        entries = [(tuple(word) + (END_OF_WORD,), count)
                   for word, count in sorted(frequencies.items()) if count >= min_freq]
        symbols = [entry[0] for entry in entries]
        weights = [entry[1] for entry in entries]
        pair_counts: Counter[tuple[str, str]] = Counter()
        pair_words: dict[tuple[str, str], set[int]] = defaultdict(set)
        for index, (parts, count) in enumerate(entries):
            for pair in zip(parts, parts[1:]):
                pair_counts[pair] += count
            for pair in self._word_pairs(parts):
                pair_words[pair].add(index)
        # Куча хранит кандидатов по частоте; одинаковые частоты разрешаем
        # лексикографическим порядком, чтобы результат был воспроизводимым.
        heap = [(-count, pair) for pair, count in pair_counts.items()]
        heapq.heapify(heap)
        for _ in range(num_merges):
            while heap:
                negative_count, pair = heapq.heappop(heap)
                if pair_counts.get(pair, 0) == -negative_count and -negative_count > 0:
                    break
            else:
                break
            self.ranks[pair] = len(self.merges)
            self.merges.append(pair)
            merged_symbol = pair[0] + pair[1]
            affected = tuple(pair_words.get(pair, ()))
            changed_pairs = set()
            for index in affected:
                previous = symbols[index]
                changed_pairs.update(self._word_pairs(previous))
                # Учитываем все вхождения пар, включая повторения внутри слова.
                for old_pair in zip(previous, previous[1:]):
                    pair_counts[old_pair] -= weights[index]
                for old_pair in self._word_pairs(previous):
                    pair_words[old_pair].discard(index)
                updated = []
                pos = 0
                while pos < len(previous):
                    if pos + 1 < len(previous) and (previous[pos], previous[pos + 1]) == pair:
                        updated.append(merged_symbol)
                        pos += 2
                    else:
                        updated.append(previous[pos])
                        pos += 1
                next_symbols = tuple(updated)
                symbols[index] = next_symbols
                for new_pair in zip(next_symbols, next_symbols[1:]):
                    pair_counts[new_pair] += weights[index]
                for new_pair in self._word_pairs(next_symbols):
                    pair_words[new_pair].add(index)
            # Обновляем в куче лишь пары, встретившиеся в изменённых словах.
            # Старые записи остаются и игнорируются при извлечении.
            for index in affected:
                changed_pairs.update(self._word_pairs(symbols[index]))
            # Удалённые пары также требуют обновления записи в куче.
            # Старые положительные значения проверяются по актуальному счётчику.
            for candidate in changed_pairs:
                count = pair_counts[candidate]
                if count > 0:
                    heapq.heappush(heap, (-count, candidate))
        return self

    def encode(self, word: str) -> list[str]:
        """Разбить слово по рангам обученных пар BPE.

        Args:
            word: Нормализованное слово.

        Returns:
            Список токенов, включая маркер конца слова в последнем токене.
        """
        parts = list(word) + [END_OF_WORD]
        while len(parts) > 1:
            candidates = [
                (self.ranks[(a, b)], index)
                for index, (a, b) in enumerate(zip(parts, parts[1:]))
                if (a, b) in self.ranks
            ]
            if not candidates:
                break
            _, index = min(candidates)
            parts[index:index + 2] = [parts[index] + parts[index + 1]]
        return parts


def build_bpe_report(words: list[str], num_merges: int) -> dict:
    """Обучить BPE и вычислить статистику по ВСЕМ словам корпуса.

    Args:
        words: Все слова исходного текста.
        num_merges: Число слияний для выбранного режима.

    Returns:
        JSON-совместимые слияния, примеры и показатели сжатия.
    """
    frequencies = Counter(words)
    tokenizer = BPETokenizer().fit(frequencies, num_merges)
    total_tokens = 0
    for word, count in frequencies.items():
        total_tokens += len(tokenizer.encode(word)) * count
    # В число букв корпуса НЕ включаем служебный маркер </w>.
    # Он необходим алгоритму слияний, но не является буквой исходного текста.
    total_chars = sum(len(word) * count for word, count in frequencies.items())
    examples = {
        word: tokenizer.encode(word)
        for word in ("наполеон", "сказала", "неудовольствие")
    }
    return {
        "num_merges": len(tokenizer.merges),
        "vocab_size": len(set(symbol for word in frequencies for symbol in word)) + 1 + len(tokenizer.merges),
        "merges": [[a, b] for a, b in tokenizer.merges],
        "stats": {
            "chars": total_chars,
            "words": len(words),
            "tokens_bpe": total_tokens,
            "tokens_per_word": total_tokens / len(words),
            "compression_vs_chars": total_chars / total_tokens,
        },
        "examples": examples,
    }


def main() -> None:
    """Подготовить корпус, обучить модели и сохранить оба файла результатов."""
    parser = argparse.ArgumentParser(description="Лабораторная работа № 1")
    parser.add_argument("--fast", action="store_true", help="быстрый режим (200 слияний)")
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
    bigram_file = ARTIFACTS / "bigram.json"
    bigram_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Перплексия для различных коэффициентов сглаживания:")
    for alpha, score in report["perplexity"].items():
        print(f"  α={alpha}: {score:.2f}")
    print(f"Результаты биграммной модели сохранены в {bigram_file}")

    number_of_merges = 200 if args.fast else 1500
    print(f"Обучение BPE: {number_of_merges} слияний...")
    bpe_report = build_bpe_report(words, number_of_merges)
    bpe_file = ARTIFACTS / "bpe.json"
    bpe_file.write_text(json.dumps(bpe_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Слияний BPE: {bpe_report['num_merges']}")
    print(f"Токенов BPE на слово: {bpe_report['stats']['tokens_per_word']:.3f}")
    print(f"Результаты BPE сохранены в {bpe_file}")


if __name__ == "__main__":
    main()
