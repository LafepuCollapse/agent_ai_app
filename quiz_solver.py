#!/usr/bin/env python3
"""OCR pytań quizowych z pamięcią odpowiedzi w SQLite."""

import argparse
import os
import re
import sqlite3
import statistics
import sys
from difflib import SequenceMatcher

import cv2
import numpy as np
import pytesseract
import requests
from PIL import Image
from pytesseract import Output

DEFAULT_DB_PATH = "quiz.sqlite3"

DEFAULT_MODEL = "google/gemini-2.5-flash-lite"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

TESSERACT_LANG = "pol"


# --------------------------------------------------------------------------
# 1-3. PREPROCESSING OBRAZU
# --------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """Wczytuje obraz jako BGR (OpenCV)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Nie mogę wczytać obrazu: {path}")
    return img


# Referencyjny kadr ekranu 1344x2992.
REFERENCE_SCREEN_HEIGHT_PX = 2992
REFERENCE_CROP_TOP_PX = 1100
REFERENCE_CROP_BOTTOM_PX = 1840
DEFAULT_CROP_TOP_FRACTION = REFERENCE_CROP_TOP_PX / \
    REFERENCE_SCREEN_HEIGHT_PX  # ≈ 0.368


def crop_middle(img: np.ndarray, crop_top_px: int = None,
                crop_top_fraction: float = None,
                crop_bottom_px: int = REFERENCE_CROP_BOTTOM_PX):
    """Zwraca pas OCR i jego offset Y."""
    h = img.shape[0]

    if crop_top_px is not None:
        cut_from = int(crop_top_px)
    else:
        fraction = crop_top_fraction if crop_top_fraction is not None else DEFAULT_CROP_TOP_FRACTION
        cut_from = int(h * fraction)

    cut_from = max(0, min(cut_from, h - 1))
    cut_to = h if crop_bottom_px is None else int(crop_bottom_px)
    cut_to = max(cut_from + 1, min(cut_to, h))
    return img[cut_from:cut_to, :], cut_from


def remove_red_circles(img: np.ndarray) -> np.ndarray:
    """Usuwa czerwone zaznaczenia przed OCR."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.dilate(red_mask, kernel, iterations=2)

    cleaned = cv2.inpaint(img, red_mask, 5, cv2.INPAINT_TELEA)
    return cleaned


OCR_UPSCALE_FACTOR = 2.0


def preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    """Skala szarości + powiększenie + threshold — poprawia jakość OCR."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, None, fx=OCR_UPSCALE_FACTOR, fy=OCR_UPSCALE_FACTOR,
                      interpolation=cv2.INTER_CUBIC)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    return thresh


def prepare_image(path: str, debug: bool = False, crop_top_px: int = None,
                  crop_top_fraction: float = None,
                  cropped_output_path: str = None,
                  crop_bottom_px: int = REFERENCE_CROP_BOTTOM_PX):
    """Zwraca obraz środkowego pasa i jego offset Y względem pełnego ekranu."""
    img = load_image(path)
    img, crop_top_offset = crop_middle(
        img, crop_top_px=crop_top_px, crop_top_fraction=crop_top_fraction,
        crop_bottom_px=crop_bottom_px)
    img = remove_red_circles(img)

    if cropped_output_path:
        cv2.imwrite(cropped_output_path, img)
    processed = preprocess_for_ocr(img)

    if debug:
        debug_path = os.path.splitext(path)[0] + "_debug.png"
        cv2.imwrite(debug_path, processed)
        print(f"[debug] Zapisano podgląd po preprocessingu: {debug_path}")

    return processed, crop_top_offset


def processed_y_to_screen_y(y_in_processed: float, crop_top_offset: int,
                            scale: float = OCR_UPSCALE_FACTOR) -> int:
    """Przelicza pozycję Y z obrazu OCR na pełny ekran."""
    return int(round(y_in_processed / scale)) + crop_top_offset


# --------------------------------------------------------------------------
# 4. OCR
# --------------------------------------------------------------------------

def run_ocr(processed_img: np.ndarray) -> str:
    pil_img = Image.fromarray(processed_img)
    text = pytesseract.image_to_string(pil_img, lang=TESSERACT_LANG)
    return text


# --------------------------------------------------------------------------
# 5. NORMALIZACJA TEKSTU
# --------------------------------------------------------------------------

def fix_common_polish_ocr_errors(text: str) -> str:
    """Poprawia częste błędy polskiego OCR."""
    # samodzielne "Ii" (np. spójnik "i") -> "i"
    text = re.sub(r"\bIi\b", "i", text)

    # końcowe 'q' w słowie (a nie samo 'q') -> 'ą'
    text = re.sub(r"(?<=[a-ząćęłńóśźżA-ZĄĆĘŁŃÓŚŹŻ])q\b", "ą", text)

    # "Pycanie"/"Pytarie"/"Pytanle" -> "Pytanie" (dla dopasowań wzorców nagłówka)
    text = re.sub(r"\bPy[tc][ae][nr][il]e\b",
                  "Pytanie", text, flags=re.IGNORECASE)

    return text


def normalize_text(raw: str) -> str:
    """Czyści wynik OCR."""
    text = raw.replace("\r", "\n")

    # Usuń znaki kontrolne poza \n
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"[|~`^*_]{2,}", "", text)

    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]  # usuń puste linie

    fixed_lines = []
    for ln in lines:
        # Typowe pomyłki OCR na początku linii oznaczającej opcję odpowiedzi,
        # np. "0)" zamiast "O)", "l." zamiast "1.", "©" zamiast "C)".
        ln = re.sub(r"^0\)", "O)", ln)
        ln = re.sub(r"^©", "C)", ln)
        ln = re.sub(r"^\$", "S)", ln)
        ln = fix_common_polish_ocr_errors(ln)
        fixed_lines.append(ln)

    return "\n".join(fixed_lines)


# Szum interfejsu.
HEADER_NOISE_PATTERNS = [
    re.compile(r"pozosta.y\s*czas", re.IGNORECASE),
    re.compile(r"^\s*pytanie\s*\d+\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*/\s*\d+\s*$"),  # np. samotne "1/5"
]


def is_header_noise(line: str) -> bool:
    return any(p.search(line) for p in HEADER_NOISE_PATTERNS)


# --------------------------------------------------------------------------
# 6. PARSOWANIE PYTANIA I ODPOWIEDZI
# --------------------------------------------------------------------------

# Rozpoznajemy opcje w formatach: "A)", "A.", "A:", "1)", "1.", "1:"
OPTION_LINE_RE = re.compile(r"^\s*([A-Da-d1-4])[\.\)\:]\s*(.+)$")


def parse_question_and_options(text: str):
    """Rozdziela pytanie i oznaczone opcje."""
    lines = text.split("\n")
    question_lines = []
    options = []

    for ln in lines:
        m = OPTION_LINE_RE.match(ln)
        if m:
            label = m.group(1).upper()
            opt_text = m.group(2).strip()
            options.append((label, opt_text))
        else:
            if not options:  # dopóki nie trafiliśmy na pierwszą opcję
                question_lines.append(ln)

    question = " ".join(question_lines).strip()
    question = re.sub(r"\s{2,}", " ", question)

    return question, options


# Parser opcji bez etykiet, oparty na odstępach między liniami.

def get_ocr_lines(pil_img: Image.Image):
    """Zwraca linie OCR z pozycjami pionowymi."""
    data = pytesseract.image_to_data(
        pil_img, lang=TESSERACT_LANG, output_type=Output.DICT
    )

    lines = []
    current_key = None
    current_words = []
    current_top = None
    current_bottom = None

    n = len(data["text"])
    for i in range(n):
        word = data["text"][i].strip()
        if not word:
            continue

        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        top = data["top"][i]
        bottom = top + data["height"][i]

        if key != current_key:
            if current_words:
                lines.append(
                    {
                        "text": " ".join(current_words),
                        "top": current_top,
                        "bottom": current_bottom,
                    }
                )
            current_key = key
            current_words = [word]
            current_top = top
            current_bottom = bottom
        else:
            current_words.append(word)
            current_top = min(current_top, top)
            current_bottom = max(current_bottom, bottom)

    if current_words:
        lines.append(
            {"text": " ".join(current_words), "top": current_top,
             "bottom": current_bottom}
        )

    lines.sort(key=lambda ln: ln["top"])
    return lines


def cluster_lines_into_blocks(lines, gap_multiplier: float = 0.6):
    """Grupuje linie według pionowych odstępów."""
    if not lines:
        return []

    heights = [ln["bottom"] - ln["top"]
               for ln in lines if ln["bottom"] > ln["top"]]
    median_height = statistics.median(heights) if heights else 20
    gap_threshold = max(median_height * gap_multiplier, 5)

    blocks = [[lines[0]]]
    for prev, curr in zip(lines, lines[1:]):
        gap = curr["top"] - prev["bottom"]
        if gap > gap_threshold:
            blocks.append([curr])
        else:
            blocks[-1].append(curr)

    block_texts = []
    for block in blocks:
        text = " ".join(ln["text"] for ln in block).strip()
        text = re.sub(r"\s{2,}", " ", text)
        text = fix_common_polish_ocr_errors(text)
        block_texts.append(text)

    return block_texts


EXPECTED_OPTIONS = 3

# Podpisy przycisków ignorowane przez OCR.
NOISE_BUTTON_TEXTS = {
    "ok", "dalej", "next", "zatwierdz", "wyslij", "kontynuuj",
    "gotowe", "zamknij", "rozumiem", "akceptuje", "akceptuję",
}


def _is_noise_button_block(text: str) -> bool:
    return _normalize_for_compare(text) in NOISE_BUTTON_TEXTS


def enforce_exact_option_count(question: str, options: list, expected: int = EXPECTED_OPTIONS):
    """Zostawia oczekiwaną liczbę opcji; nadmiar dołącza do pytania."""
    if len(options) <= expected:
        return question, options

    overflow = options[:-expected]
    kept = options[-expected:]

    extra_text = " ".join(text for _, text in overflow).strip()
    merged_question = " ".join(part for part in (
        question, extra_text) if part).strip()
    merged_question = re.sub(r"\s{2,}", " ", merged_question)

    renumbered = [(str(i), text) for i, (_, text) in enumerate(kept, start=1)]
    return merged_question, renumbered


def parse_question_and_options_by_layout(pil_img: Image.Image, gap_multiplier: float = 0.6):
    """Parsuje pytanie i nieoznaczone opcje według układu."""
    lines = get_ocr_lines(pil_img)
    text_lines = [fix_common_polish_ocr_errors(
        ln["text"].strip()) for ln in lines]

    filtered = [ln for ln, txt in zip(
        lines, text_lines) if txt and not is_header_noise(txt)]

    blocks = cluster_lines_into_blocks(filtered, gap_multiplier=gap_multiplier)
    blocks = [b for b in blocks if b]
    blocks = [b for b in blocks if not _is_noise_button_block(b)]

    if len(blocks) < 2:
        return "", []

    question_end_idx = None
    for i, block in enumerate(blocks):
        if block.rstrip().endswith("?"):
            question_end_idx = i
            break

    if question_end_idx is not None:
        question = " ".join(blocks[: question_end_idx + 1]).strip()
        question = re.sub(r"\s{2,}", " ", question)
        option_blocks = blocks[question_end_idx +
                               1: question_end_idx + 1 + EXPECTED_OPTIONS]
        options = [(str(i), text)
                   for i, text in enumerate(option_blocks, start=1)]
        return question, options

    # Fallback bez znaku zapytania.
    question = blocks[0]
    options = [(str(i), text) for i, text in enumerate(blocks[1:], start=1)]
    return question, options


# Etykieta OCR na początku opcji.
_LEADING_LABEL_RE = re.compile(r"^\s*[A-Da-d1-4][\.\)\:]\s*")


def locate_option_bounds(pil_img: Image.Image, options, gap_multiplier: float = 0.6):
    """Zwraca granice opcji na obrazie OCR albo None."""
    lines = get_ocr_lines(pil_img)

    prepared = []
    for ln in lines:
        text = fix_common_polish_ocr_errors(ln["text"].strip())
        text = _LEADING_LABEL_RE.sub("", text)
        if not text or is_header_noise(text):
            continue
        prepared.append(
            {"text": text, "top": ln["top"], "bottom": ln["bottom"]})

    bounds = []
    for _, opt_text in options:
        norm_opt = _normalize_for_compare(opt_text)
        matched = [
            ln for ln in prepared
            if len(_normalize_for_compare(ln["text"])) >= 3
            and _normalize_for_compare(ln["text"]) in norm_opt
        ]
        bounds.append(
            (min(l["top"] for l in matched), max(l["bottom"] for l in matched))
            if matched else None
        )

    return bounds


# --------------------------------------------------------------------------
# 7. BAZA SQLITE
# --------------------------------------------------------------------------

def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(answers)")
    }
    if {"id", "source", "created_at"} & columns:
        conn.execute("PRAGMA foreign_keys = OFF;")
        conn.executescript(
            """
            BEGIN;
            CREATE TABLE questions_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_text TEXT NOT NULL UNIQUE
            );
            INSERT INTO questions_new (id, question_text)
                SELECT id, question_text FROM questions;
            CREATE TABLE answers_new (
                question_id INTEGER NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
                answer_text TEXT NOT NULL
            );
            INSERT INTO answers_new (question_id, answer_text)
                SELECT question_id, answer_text FROM answers;
            DROP TABLE options;
            DROP TABLE answers;
            DROP TABLE questions;
            ALTER TABLE questions_new RENAME TO questions;
            ALTER TABLE answers_new RENAME TO answers;
            COMMIT;
            """
        )
        conn.execute("PRAGMA foreign_keys = ON;")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question_text TEXT NOT NULL UNIQUE
        );

        CREATE TABLE IF NOT EXISTS answers (
            question_id INTEGER NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
            answer_text TEXT NOT NULL
        );
        """
    )
    conn.commit()
    return conn


def get_question_id(conn: sqlite3.Connection, question_text: str):
    row = conn.execute(
        "SELECT id FROM questions WHERE question_text = ?", (question_text,)
    ).fetchone()
    return row[0] if row else None


def save_question(conn: sqlite3.Connection, question_text: str):
    cur = conn.execute(
        "INSERT INTO questions (question_text) VALUES (?)",
        (question_text,),
    )
    conn.commit()
    return cur.lastrowid


def get_saved_answers(conn: sqlite3.Connection, question_id: int):
    return conn.execute(
        "SELECT answer_text FROM answers WHERE question_id = ? ORDER BY answer_text",
        (question_id,),
    ).fetchall()


def save_answer(conn: sqlite3.Connection, question_id: int, answer_text: str):
    conn.execute(
        "INSERT INTO answers (question_id, answer_text) VALUES (?, ?)",
        (question_id, answer_text),
    )
    conn.commit()


# --------------------------------------------------------------------------
# 8. ZAPYTANIE DO OPENROUTER (LLM)
# --------------------------------------------------------------------------

def ask_llm(question: str, options, model: str = DEFAULT_MODEL) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "Brak OPENROUTER_API_KEY w zmiennych środowiskowych. "
            "Ustaw: export OPENROUTER_API_KEY='sk-or-...'"
        )

    options_str = "\n".join(f"{label}) {text}" for label, text in options)

    prompt = (
        "Poniżej jest pytanie quizowe wielokrotnego wyboru po polsku wraz z opcjami "
        "odpowiedzi. Podaj TYLKO pełną treść poprawnej opcji, bez jej numeru ani litery "
        "i bez dodatkowych wyjaśnień po polsku. Jeśli nie masz pewności, wybierz "
        "najbardziej prawdopodobną odpowiedź.\n\n"
        f"Pytanie: {question}\n\n"
        f"Opcje:\n{options_str}\n\n"
        "Odpowiedz w formacie: <treść opcji>"
    )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    resp = requests.post(OPENROUTER_URL, json=payload,
                         headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


# --------------------------------------------------------------------------
# 9. DOPASOWANIE ODPOWIEDZI DO NUMERU OPCJI
# --------------------------------------------------------------------------

def _normalize_for_compare(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


_ANSWER_PREFIX_RE = re.compile(r"^\s*(?:[A-Da-d]|[1-4])[\.\)\:]\s*")


def clean_answer_for_storage(answer_text: str) -> str:
    """Usuwa numer/literę opcji, aby baza przechowywała wyłącznie jej treść."""
    return _ANSWER_PREFIX_RE.sub("", answer_text).strip()


def answer_match_percent(answer_text: str, option_text: str) -> float:
    """Zwraca podobieństwo treści odpowiedzi i opcji w skali 0--100."""
    answer = _normalize_for_compare(clean_answer_for_storage(answer_text))
    option = _normalize_for_compare(option_text)
    if not answer or not option:
        return 0.0
    if answer in option or option in answer:
        return 100.0

    sequence = SequenceMatcher(None, answer, option).ratio()
    answer_words, option_words = set(answer.split()), set(option.split())
    overlap = len(answer_words & option_words) / len(answer_words | option_words)
    return round(max(sequence, overlap) * 100, 1)


def select_best_saved_answer(saved_answers, options):
    """Wybiera wpis bazy najbliższy którejkolwiek bieżącej opcji odpowiedzi."""
    return max(
        saved_answers,
        key=lambda row: max(
            (answer_match_percent(row[0], option_text)
             for _, option_text in options), default=0.0),
    )


def match_answer_to_option(answer_text: str, options):
    """Zwraca najlepiej dopasowaną opcję i wynik 0--100."""
    best_idx, best_label, best_text, best_score = None, None, None, -1.0

    for i, (opt_label, opt_text) in enumerate(options, start=1):
        score = answer_match_percent(answer_text, opt_text)
        if score > best_score:
            best_idx, best_label, best_text, best_score = i, opt_label, opt_text, score

    if best_idx is not None:
        return best_idx, best_label, best_text, best_score

    return None, None, None, 0.0


# --------------------------------------------------------------------------
# API DO UŻYCIA Z INNYCH SKRYPTÓW (np. scenario_runner.py)
# --------------------------------------------------------------------------

def solve_quiz_from_image(image_path: str, db_path: str = DEFAULT_DB_PATH,
                          model: str = DEFAULT_MODEL, gap_multiplier: float = 0.6,
                          debug: bool = False, crop_top_px: int = None,
                          crop_top_fraction: float = None,
                          cropped_output_path: str = None,
                          crop_bottom_px: int = REFERENCE_CROP_BOTTOM_PX) -> dict:
    """Rozwiązuje quiz ze zrzutu i zwraca wynik jako dict."""
    processed, crop_top_offset = prepare_image(
        image_path, debug=debug, crop_top_px=crop_top_px,
        crop_top_fraction=crop_top_fraction,
        cropped_output_path=cropped_output_path,
        crop_bottom_px=crop_bottom_px,
    )
    pil_processed = Image.fromarray(processed)
    raw_text = run_ocr(processed)
    clean_text = normalize_text(raw_text)

    question, options = parse_question_and_options(clean_text)
    used_layout_parser = False
    if not question or not options or len(options) < EXPECTED_OPTIONS:
        question, options = parse_question_and_options_by_layout(
            pil_processed, gap_multiplier=gap_multiplier
        )
        used_layout_parser = True

    question, options = enforce_exact_option_count(question, options)

    result = {
        "question": question,
        "options": options,
        "used_layout_parser": used_layout_parser,
        "raw_text": raw_text,
        "clean_text": clean_text,
        "chosen_answer_text": None,
        "idx": None,
        "label": None,
        "option_text": None,
        "source": None,
        "match_percent": None,
        "is_new_question": None,
        "answer_y": None,
    }

    if not question or not options:
        return result

    conn = init_db(db_path)
    try:
        question_id = get_question_id(conn, question)
        if question_id is None:
            question_id = save_question(conn, question)
            is_new_question = True
        else:
            is_new_question = False

        saved_answers = get_saved_answers(conn, question_id)
        if saved_answers:
            best_saved = select_best_saved_answer(saved_answers, options)
            chosen_answer_text = best_saved[0]
            source = "db"
        else:
            chosen_answer_text = ask_llm(question, options, model=model)
            chosen_answer_text = clean_answer_for_storage(chosen_answer_text)
            save_answer(conn, question_id, chosen_answer_text)
            source = f"llm:{model}"

        idx, label, opt_text, match_percent = match_answer_to_option(
            chosen_answer_text, options)

        answer_y = None
        if idx is not None:
            option_bounds = locate_option_bounds(
                pil_processed, options, gap_multiplier=gap_multiplier)
            chosen_bounds = (
                option_bounds[idx - 1] if idx -
                1 < len(option_bounds) else None
            )
            if chosen_bounds:
                top, bottom = chosen_bounds
                answer_y = processed_y_to_screen_y(
                    (top + bottom) / 2, crop_top_offset)

        result.update({
            "chosen_answer_text": chosen_answer_text,
            "idx": idx,
            "label": label,
            "option_text": opt_text,
            "source": source,
            "match_percent": match_percent,
            "is_new_question": is_new_question,
            "answer_y": answer_y,
        })
    finally:
        conn.close()

    return result


# --------------------------------------------------------------------------
# GŁÓWNY PRZEPŁYW
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="OCR quizu + baza SQLite + OpenRouter LLM")
    parser.add_argument("image", help="Ścieżka do zdjęcia ekranu z pytaniem")
    parser.add_argument("--db", default=DEFAULT_DB_PATH,
                        help="Ścieżka do bazy SQLite")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Model OpenRouter do użycia")
    parser.add_argument("--debug", action="store_true",
                        help="Zapisz podgląd obrazu po preprocessingu")
    parser.add_argument(
        "--gap-multiplier",
        type=float,
        default=0.6,
        help="Czułość grupowania linii w opcje bez etykiet (mniejsza wartość = łatwiej rozdziela bloki)",
    )
    parser.add_argument(
        "--crop-top-px", type=int, default=None,
        help="Sztywne cięcie góry zrzutu w pikselach (np. 1100), żeby ominąć "
             "graficzny nagłówek 'Pytanie N', który OCR czyta jako śmieci.",
    )
    parser.add_argument(
        "--crop-top-fraction", type=float, default=None,
        help="Cięcie góry zrzutu jako proporcja wysokości (0-1); używane, gdy "
             "nie podano --crop-top-px. Domyślnie ok. 0.368 (odpowiada 1100px "
             "na zrzucie 1344x2992).",
    )
    parser.add_argument(
        "--crop-bottom-px", type=int, default=REFERENCE_CROP_BOTTOM_PX,
        help="Dolna granica pasa OCR w pikselach pełnego zrzutu (domyślnie 1840).",
    )
    args = parser.parse_args()

    # 1-4. Obraz -> OCR
    processed, crop_top_offset = prepare_image(
        args.image, debug=args.debug,
        crop_top_px=args.crop_top_px,
        crop_top_fraction=args.crop_top_fraction,
        crop_bottom_px=args.crop_bottom_px,
    )
    pil_processed = Image.fromarray(processed)
    raw_text = run_ocr(processed)

    # 5. Normalizacja
    clean_text = normalize_text(raw_text)
    print("----- Tekst po OCR i normalizacji -----")
    print(clean_text)
    print("----------------------------------------")

    # 6. Parsowanie — najpierw próba z jawnymi etykietami (A)/B)/1./2. itd.),
    # a jeśli quiz ich nie ma (typowe dla przycisków bez numeracji),
    # przechodzimy na parsowanie oparte na układzie (odstępach pionowych).
    question, options = parse_question_and_options(clean_text)
    used_layout_parser = False

    if not question or not options or len(options) < EXPECTED_OPTIONS:
        question, options = parse_question_and_options_by_layout(
            pil_processed, gap_multiplier=args.gap_multiplier
        )
        used_layout_parser = True

    question, options = enforce_exact_option_count(question, options)

    if not question or not options:
        print("Nie udało się wyodrębnić pytania i/lub opcji odpowiedzi z obrazu.")
        print(f"Pytanie: {question!r}")
        print(f"Opcje: {options!r}")
        print(
            "Wskazówka: spróbuj dostroić --gap-multiplier (np. 0.4 albo 0.8) "
            "albo sprawdź podgląd z --debug."
        )
        sys.exit(1)

    if used_layout_parser:
        print("(Użyto parsera opartego na układzie — quiz bez etykiet opcji.)")

    print(f"\nRozpoznane pytanie: {question}")
    print("Rozpoznane opcje:")
    for label, text in options:
        print(f"  {label}) {text}")

    # 7. Baza
    conn = init_db(args.db)
    question_id = get_question_id(conn, question)

    if question_id is None:
        print("\nPytania nie ma jeszcze w bazie — dodaję.")
        question_id = save_question(conn, question)
    else:
        print("\nPytanie już istnieje w bazie.")

    saved_answers = get_saved_answers(conn, question_id)

    if saved_answers:
        print(
            f"Znaleziono {len(saved_answers)} zapisaną/zapisane odpowiedź(zi) w bazie:")
        for (ans_text,) in saved_answers:
            print(f"  - {ans_text}")
        # Wybieramy wpis najbardziej podobny do bieżących opcji (kolejność
        # odpowiedzi na ekranie może się zmieniać między podejściami).
        chosen_answer_text = select_best_saved_answer(saved_answers, options)[0]
    else:
        # 8. Brak odpowiedzi w bazie -> pytamy LLM
        print("\nBrak zapisanej odpowiedzi — pytam model przez OpenRouter...")
        chosen_answer_text = ask_llm(question, options, model=args.model)
        print(f"Odpowiedź modelu: {chosen_answer_text}")
        chosen_answer_text = clean_answer_for_storage(chosen_answer_text)
        save_answer(conn, question_id, chosen_answer_text)

    # 9. Dopasowanie do numeru opcji
    idx, label, opt_text, match_percent = match_answer_to_option(
        chosen_answer_text, options)

    print("\n========== WYNIK ==========")
    if idx is not None:
        print(
            f"Poprawna odpowiedź to opcja nr {idx} ({label}): {opt_text} "
            f"(dopasowanie {match_percent:.1f}%)"
        )

        option_bounds = locate_option_bounds(
            pil_processed, options, gap_multiplier=args.gap_multiplier)
        chosen_bounds = (
            option_bounds[idx - 1] if idx - 1 < len(option_bounds) else None
        )
        if chosen_bounds:
            top, bottom = chosen_bounds
            answer_y = processed_y_to_screen_y(
                (top + bottom) / 2, crop_top_offset)
            print(f"Pozycja Y na zrzucie ekranu: {answer_y}")
        else:
            print(
                "[uwaga] Nie udało się zlokalizować bounding boxa tej opcji.")
    else:
        print("Nie udało się jednoznacznie dopasować odpowiedzi do żadnej z opcji.")
        print(f"Surowa odpowiedź: {chosen_answer_text}")

    conn.close()


if __name__ == "__main__":
    main()
