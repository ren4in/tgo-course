# ============================================================================
# Файл: run.py
# Создан: 03.11.2026
# Автор: Измайлов С.В.
# Проект: Технологии глубокого обучения и нейронных сетей, работа pt01
# Описание: Загрузка корпуса «Война и мир», нормализация и выделение слов,
#           разделение данных на обучающую и проверочную части, расчёт частот.
# ============================================================================

"""Загрузка, нормализация и статистический анализ текстового корпуса."""

import argparse
from collections import Counter
from pathlib import Path
import re
from urllib.request import urlopen


# Все пути задаются относительно самого скрипта: запуск возможен из любой папки.
HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
CORPUS = HERE / "corpus.txt"

# Фиксированный корпус из задания. Текст загружается только при отсутствии файла.
CORPUS_URL = (
    "https://raw.githubusercontent.com/nevmenandr/word2vec-russian-novels/"
    "master/books_before/WarAndPeace.txt"
)

# Словом считается последовательность русских букв, допускающая дефис внутри.
# Знаки препинания, цифры и другие символы в последовательность слов не входят.
WORD_PATTERN = re.compile(r"[а-я]+(?:-[а-я]+)*")


def load_corpus() -> str:
    """Загрузить корпус из локального файла или скачать его при первом запуске.

    Returns:
        Полный исходный текст романа в виде строки.
    """
    if not CORPUS.exists():
        # Сохраняем исходные байты: при повторных запусках сеть не требуется.
        print("Скачивание корпуса «Война и мир»...")
        with urlopen(CORPUS_URL, timeout=60) as response:
            content = response.read()
        CORPUS.write_bytes(content)

    # utf-8-sig также корректно обрабатывает возможную метку BOM в начале файла.
    return CORPUS.read_text(encoding="utf-8-sig")


def tokenize(text: str) -> list[str]:
    """Привести текст к единому виду и извлечь русские слова.

    Args:
        text: Исходный текст, содержащий слова и знаки препинания.

    Returns:
        Список слов в исходном порядке, в нижнем регистре и без буквы «ё».
    """
    # Нормализация обеспечивает одинаковый подсчёт слов «Её» и «ее».
    normalized = text.lower().replace("ё", "е")
    return WORD_PATTERN.findall(normalized)


def split_words(words: list[str]) -> tuple[list[str], list[str]]:
    """Разделить последовательность слов на обучение (90%) и проверку (10%).

    Args:
        words: Полный список токенизированных слов корпуса.

    Returns:
        Пара списков: обучающая и проверочная части соответственно.
    """
    # Важно сохранять порядок: последние 10% слов модель не увидит при обучении.
    split_index = int(len(words) * 0.9)
    return words[:split_index], words[split_index:]


def main() -> None:
    """Подготовить данные и напечатать основные статистические показатели."""
    # Параметр --fast предусмотрен интерфейсом запуска лабораторной работы.
    # При подготовке корпуса оба режима используют одинаковую обработку текста.
    parser = argparse.ArgumentParser(description="Лабораторная работа № 1")
    parser.add_argument("--fast", action="store_true", help="быстрый режим для CI")
    args = parser.parse_args()

    # Каталог результатов располагается рядом со скриптом.
    ARTIFACTS.mkdir(exist_ok=True)

    # Текст последовательно проходит загрузку, нормализацию и разделение.
    words = tokenize(load_corpus())
    train_words, val_words = split_words(words)

    # Counter хранит абсолютную частоту каждого слова в обучающей части.
    frequencies = Counter(train_words)

    # Выводим показатели для ручной проверки корректности подготовки корпуса.
    print(f"Режим: {'быстрый' if args.fast else 'полный'}")
    print(f"Всего слов: {len(words)}")
    print(f"Обучающих слов: {len(train_words)}")
    print(f"Проверочных слов: {len(val_words)}")
    print(f"Размер словаря обучения: {len(frequencies)}")
    print("20 самых частых слов обучающей части:")
    for word, count in frequencies.most_common(20):
        print(f"  {word}: {count}")

    print("Обработка корпуса завершена.")


if __name__ == "__main__":
    main()
