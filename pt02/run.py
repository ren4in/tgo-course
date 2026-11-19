# ============================================================================
# Файл: run.py
# Создан: 19.11.2026
# Автор: Измайлов С.В.
# Проект: Технологии глубокого обучения и нейронных сетей, работа pt02
# Описание: Первый этап второй лабораторной: подготовка корпуса и применение
#           BPE-токенизатора, обученного в первой лабораторной работе.
# ============================================================================
"""Подготовить поток BPE-токенов для будущего обучения трансформера."""

import argparse
import json
from pathlib import Path
import re
from urllib.request import urlopen


# Путь к файлам определяется расположением скрипта, а не текущей папкой консоли.
HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
CORPUS = HERE.parent / "pt01" / "corpus.txt"
BPE_FILE = HERE.parent / "pt01" / "artifacts" / "bpe.json"
CORPUS_URL = (
    "https://raw.githubusercontent.com/nevmenandr/word2vec-russian-novels/"
    "master/books_before/WarAndPeace.txt"
)
WORD_PATTERN = re.compile(r"[а-я]+(?:-[а-я]+)*")
END_OF_WORD = "</w>"


def load_words() -> tuple[list[str], list[str]]:
    """Загрузить роман, нормализовать слова и разделить их в отношении 90/10.

    Returns:
        Обучающие слова и последние десять процентов проверочных слов.
    """
    if not CORPUS.exists():
        # Если после первой работы корпус удалён, скачиваем его повторно.
        CORPUS.parent.mkdir(parents=True, exist_ok=True)
        print("Скачивание корпуса «Война и мир»...")
        with urlopen(CORPUS_URL, timeout=60) as response:
            CORPUS.write_bytes(response.read())

    source = CORPUS.read_text(encoding="utf-8-sig")
    normalized = source.lower().replace("ё", "е")
    words = WORD_PATTERN.findall(normalized)
    border = int(0.9 * len(words))

    # Порядок слов сохраняем: проверка должна быть на конце романа,
    # а не на случайных словах, которые могли встретиться в обучении.
    return words[:border], words[border:]


class BPETokenizer:
    """Применять готовые слияния BPE из первой работы по их порядку."""

    def __init__(self, bpe_file: Path, words: list[str]) -> None:
        """Восстановить список слияний, их ранги и номера токенов.

        Args:
            bpe_file: JSON с обученными в первой работе слияниями.
            words: Слова корпуса для определения исходного алфавита.
        """
        report = json.loads(bpe_file.read_text(encoding="utf-8"))
        self.merges: list[tuple[str, str]] = [
            (left, right) for left, right in report["merges"]
        ]
        self.ranks = {pair: index for index, pair in enumerate(self.merges)}

        # Словарь нужен нейросети: она работает с целыми числами, не со строками.
        # Сначала символы и маркер конца слова, затем результаты всех слияний.
        alphabet = sorted(set("".join(words)))
        self.id_to_token = alphabet + [END_OF_WORD] + [
            left + right for left, right in self.merges
        ]
        if len(self.id_to_token) != len(set(self.id_to_token)):
            raise ValueError("В словаре BPE обнаружены повторяющиеся токены")
        self.token_to_id = {
            token: index for index, token in enumerate(self.id_to_token)
        }
        self.cache: dict[str, list[int]] = {}

    def encode_word(self, word: str) -> list[int]:
        """Превратить слово в номера BPE-токенов, учитывая ранги слияний.

        Args:
            word: Нормализованное русское слово.

        Returns:
            Номера токенов в исходном порядке, включая конец слова.
        """
        if word in self.cache:
            return self.cache[word]

        pieces = list(word) + [END_OF_WORD]
        while len(pieces) > 1:
            # Из доступных соседних пар сначала сливается самая ранняя
            # в обученном списке BPE, а не самая длинная строка.
            choices = [
                (self.ranks[(left, right)], index)
                for index, (left, right) in enumerate(zip(pieces, pieces[1:]))
                if (left, right) in self.ranks
            ]
            if not choices:
                break
            _, index = min(choices)
            pieces[index:index + 2] = [pieces[index] + pieces[index + 1]]

        ids = [self.token_to_id[piece] for piece in pieces]
        self.cache[word] = ids
        return ids

    def encode_words(self, words: list[str]) -> list[int]:
        """Объединить слова в единый поток номеров токенов для модели."""
        return [token_id for word in words for token_id in self.encode_word(word)]

    def decode(self, ids: list[int]) -> str:
        """Собрать текст обратно, заменив маркеры конца слова пробелами."""
        pieces = (self.id_to_token[index] for index in ids)
        return "".join(pieces).replace(END_OF_WORD, " ").strip()


def inspect_data(bpe_file: Path) -> None:
    """Показать результат первого этапа, не запуская обучение модели."""
    train_words, val_words = load_words()
    tokenizer = BPETokenizer(bpe_file, train_words + val_words)
    train_ids = tokenizer.encode_words(train_words)
    val_ids = tokenizer.encode_words(val_words)
    example = train_words[:5]
    example_ids = tokenizer.encode_words(example)

    print(f"Обучающих слов: {len(train_words)}")
    print(f"Проверочных слов: {len(val_words)}")
    print(f"Слияний BPE: {len(tokenizer.merges)}")
    print(f"Размер словаря токенов: {len(tokenizer.id_to_token)}")
    print(f"Токенов для обучения: {len(train_ids)}")
    print(f"Токенов для проверки: {len(val_ids)}")
    print(f"Пример слов: {' '.join(example)}")
    print(f"Номера токенов: {example_ids}")
    print(f"Обратная сборка: {tokenizer.decode(example_ids)}")


def main() -> None:
    """Разобрать параметры запуска и выполнить доступный этап работы."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="быстрый прогон (позже)")
    parser.add_argument("--bpe", type=Path, default=BPE_FILE, help="путь к bpe.json")
    parser.add_argument(
        "--inspect-data", action="store_true",
        help="проверить подготовку текста и BPE без обучения",
    )
    args = parser.parse_args()

    if args.inspect_data:
        inspect_data(args.bpe)
        return

    # Заглушку уберём после реализации остальных этапов. Пока она здесь,
    # CI не принимает частичную реализацию за готовую лабораторную.
    raise NotImplementedError("Обучение трансформера будет добавлено позже")


if __name__ == "__main__":
    main()
