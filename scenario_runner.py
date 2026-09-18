#!/usr/bin/env python3
"""Wykonuje scenariusz JSON przez ADB, OCR i LLM.

Użycie: python3 scenario_runner.py scenario.json
Akcje: tap_text, tap_xy, type_text, press_enter, wait, wait_random,
wait_input_field_and_type, wait_webview, checkbox_tap, screenshot,
solve_and_tap, repeat, log.
"""

import argparse
import glob
import json
import os
import random
import sys
import time
from datetime import datetime

import android as phone
import quiz_solver as quiz


def load_scenario(path: str):
    """Wczytuje listę kroków lub obiekt config/steps."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return {}, data

    if isinstance(data, dict):
        config = data.get("config", {})
        steps = data.get("steps")
        if steps is None:
            raise ValueError(
                "Scenariusz w formie obiektu musi mieć klucz 'steps' "
                "(listę kroków)."
            )
        return config, steps

    raise ValueError(
        "Scenariusz musi być albo listą kroków, albo obiektem "
        "{'config': {...}, 'steps': [...]}."
    )


def auto_screenshot_path(ctx: dict) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return os.path.join(ctx["screenshot_dir"], f"screen_{ts}.png")


def clear_previous_quiz_images(screenshot_dir: str):
    """Usuwa automatyczne obrazy poprzedniego quizu."""
    patterns = ("screen_*.png", "last_ocr_crop.png")
    for pattern in patterns:
        for path in glob.glob(os.path.join(screenshot_dir, pattern)):
            os.remove(path)


def substitute_step(value, ctx):
    """Podstawia {code} w wartościach kroku."""
    code = ctx.get("code")

    if isinstance(value, str):
        if "{code}" in value:
            if code is None:
                raise RuntimeError(
                    "Krok zawiera placeholder '{code}', ale nie podano kodu — "
                    "użyj --code TWOJKOD albo dodaj \"code\": \"...\" w "
                    "sekcji 'config' scenariusza."
                )
            return value.replace("{code}", code)
        return value

    if isinstance(value, dict):
        return {k: substitute_step(v, ctx) for k, v in value.items()}

    if isinstance(value, list):
        return [substitute_step(v, ctx) for v in value]

    return value


def h_log(ctx: dict, step: dict):
    print(f"  {step.get('message', '')}")


def h_tap_text(ctx: dict, step: dict):
    phone.wait_for_and_click_text(
        ctx["device"], step["text"],
        step.get("timeout", 15.0), step.get("poll_interval", 0.5),
    )


def h_tap_xy(ctx: dict, step: dict):
    phone.tap(ctx["device"], int(step["x"]), int(step["y"]))


def h_type_text(ctx: dict, step: dict):
    phone.type_text(ctx["device"], str(step["text"]))


def h_press_enter(ctx: dict, step: dict):
    phone.press_enter(ctx["device"])


def h_wait(ctx: dict, step: dict):
    time.sleep(float(step["seconds"]))


def h_wait_random(ctx: dict, step: dict):
    delay = random.uniform(float(step.get("min", 0.5)),
                           float(step.get("max", 1.5)))
    print(f"  (czekam losowo {delay:.2f}s)")
    time.sleep(delay)


def h_wait_input_field_and_type(ctx: dict, step: dict):
    hint_words = tuple(step.get("hint_words", ("kod", "code")))
    phone.wait_for_input_field_and_type(
        ctx["device"], str(step["text"]),
        step.get("timeout", 10.0), step.get("poll_interval", 0.5),
        hint_words=hint_words,
    )


def h_wait_webview(ctx: dict, step: dict):
    phone.wait_for_webview(
        ctx["device"], step.get("timeout", 20.0), step.get(
            "poll_interval", 0.5)
    )


def h_checkbox_tap(ctx: dict, step: dict):
    phone.tap_checkbox_fixed(
        ctx["device"], int(step["x"]), int(step["y"]),
        step.get("delay_min", 0.3), step.get("delay_max", 1.2),
    )


def h_screenshot(ctx: dict, step: dict):
    path = step.get("path") or auto_screenshot_path(ctx)
    phone.take_screenshot(ctx["device"], path)
    ctx["last_screenshot"] = path
    print(f"  -> zapisano zrzut: {path}")


def h_solve_and_tap(ctx: dict, step: dict):
    wait_min = step.get("wait_before_min")
    wait_max = step.get("wait_before_max")

    if wait_min is not None and wait_max is not None:
        delay = random.uniform(float(wait_min), float(wait_max))
        print(f"  (czekam losowo {delay:.2f}s przed zrzutem)")
        time.sleep(delay)

    path = step.get("screenshot_path") or auto_screenshot_path(ctx)
    phone.take_screenshot(ctx["device"], path)
    ctx["last_screenshot"] = path
    print(f"  -> zrzut do OCR: {path}")

    result = quiz.solve_quiz_from_image(
        path,
        db_path=step.get(
            "db",
            ctx["config"].get("db", quiz.DEFAULT_DB_PATH)
        ),
        model=step.get(
            "model",
            ctx["config"].get("model", quiz.DEFAULT_MODEL)
        ),
        gap_multiplier=step.get(
            "gap_multiplier",
            ctx["config"].get("gap_multiplier", 0.6)
        ),
        cropped_output_path=os.path.join(
            ctx["screenshot_dir"], "last_ocr_crop.png"),
        crop_bottom_px=step.get(
            "crop_bottom_px", ctx["config"].get("crop_bottom_px", 1840)),
    )

    ctx["last_quiz_result"] = result

    print(f"  Pytanie: {result['question']!r}")

    for label, text in result["options"]:
        print(f"    {label}) {text}")

    match_percent = result.get("match_percent")
    match_info = (f" ({match_percent:.1f}% dopasowania)"
                  if match_percent is not None else "")
    print(
        f"  Odpowiedź ({result['source']}): "
        f"{result['chosen_answer_text']!r} "
        f"-> opcja {result['idx']}{match_info}"
    )

    idx = result["idx"]

    if idx is None:
        print(
            "  [uwaga] Nie udało się dopasować odpowiedzi "
            "do żadnej opcji — pomijam kliknięcie."
        )

        if step.get("required", False):
            raise RuntimeError(
                "solve_and_tap: brak dopasowania odpowiedzi, required=true"
            )

        return

    # Pozycję opcji wyznacza OCR; WebView nie udostępnia jej w drzewie UI.
    answer_y = result.get("answer_y")

    if answer_y is None:
        option_coords = step.get(
            "option_coords", ctx["config"].get("option_coords", {}))
        fallback = option_coords.get(str(idx))
        if not fallback or len(fallback) != 2:
            raise RuntimeError(
                "solve_and_tap: OCR nie ustalił pozycji tekstu, a dla opcji "
                f"{idx} nie podano domyślnej pozycji w option_coords."
            )
        x, y = map(int, fallback)
        print(f"  [uwaga] Brak pozycji z OCR — używam domyślnej: ({x}, {y})")
    else:
        x = int(step.get("x", 125))
        y = answer_y
        print(f"  Pozycja odpowiedzi z OCR: y={y} (x stałe: {x})")

    delay = random.uniform(
        float(step.get("tap_delay_min", 0.3)),
        float(step.get("tap_delay_max", 1.0))
    )

    print(
        f"  Klikam w opcję {idx} "
        f"({x}, {y}) po {delay:.2f}s..."
    )

    time.sleep(delay)

    phone.tap(
        ctx["device"],
        int(x),
        int(y)
    )


def h_repeat(ctx: dict, step: dict):
    times = int(step.get("times", 1))
    inner_steps = step.get("steps", [])
    for i in range(times):
        print(f"\n-- powtórzenie {i + 1}/{times} --")
        run_steps(ctx, inner_steps)


ACTIONS = {
    "log": h_log,
    "tap_text": h_tap_text,
    "tap_xy": h_tap_xy,
    "type_text": h_type_text,
    "press_enter": h_press_enter,
    "wait": h_wait,
    "wait_random": h_wait_random,
    "wait_input_field_and_type": h_wait_input_field_and_type,
    "wait_webview": h_wait_webview,
    "checkbox_tap": h_checkbox_tap,
    "screenshot": h_screenshot,
    "solve_and_tap": h_solve_and_tap,
    "repeat": h_repeat,
}


def run_steps(ctx: dict, steps: list):
    for i, step in enumerate(steps, start=1):
        step = substitute_step(step, ctx)
        action = step.get("action")
        handler = ACTIONS.get(action)
        print(f"\n[{i}/{len(steps)}] {action}")

        if handler is None:
            raise ValueError(
                f"Nieznana akcja: {action!r} (krok #{i}). Dostępne akcje: "
                f"{sorted(ACTIONS.keys())}"
            )

        if ctx.get("dry_run"):
            print(f"  (dry-run) {step}")
            continue

        try:
            handler(ctx, step)
        except Exception as exc:
            if step.get("continue_on_error"):
                print(f"  [błąd, ale continue_on_error=true, idę dalej] {exc}")
                continue
            raise RuntimeError(f"Błąd w kroku #{i} ({action}): {exc}") from exc


def main():
    parser = argparse.ArgumentParser(
        description="Wykonuje scenariusz (lista kroków w JSON) łączący sterowanie "
                    "telefonem przez ADB i rozwiązywanie quizów (OCR + LLM)."
    )
    parser.add_argument("scenario", help="Ścieżka do pliku scenariusza (JSON)")
    parser.add_argument("--device", default=None,
                        help="Numer seryjny urządzenia (adb -s ...)")
    parser.add_argument("--connect", default=None,
                        help="Adres do 'adb connect' (np. 192.168.1.50:5555)")
    parser.add_argument("--adb-path", default=None,
                        help="Jawna ścieżka do binarki adb, gdy nie jest w PATH")
    parser.add_argument("--screenshot-dir", default=".",
                        help="Katalog na zrzuty bieżącego quizu; poprzednie screen_*.png są czyszczone na starcie")
    parser.add_argument("--dry-run", action="store_true",
                        help="Tylko wypisz kroki, bez wykonywania czegokolwiek na telefonie")
    parser.add_argument("--code", default=None,
                        help="Kod podstawiany za placeholder '{code}' w krokach scenariusza "
                             "(np. w polu 'text' kroku wait_input_field_and_type/type_text). "
                             "Jeśli nie podane, brany jest 'config.code' ze scenariusza.")
    args = parser.parse_args()

    config, steps = load_scenario(args.scenario)

    if not args.dry_run:
        if args.adb_path:
            phone.resolve_adb_binary(
                explicit_path=os.path.expanduser(args.adb_path))
        else:
            phone.resolve_adb_binary()
        print(f"Używam adb: {phone._ADB_BIN}")

        connect_addr = args.connect or config.get("connect")
        if connect_addr:
            phone.adb_connect(connect_addr)

        device = phone.ensure_device_available(
            args.device or config.get("device"))
        print(f"Używam urządzenia: {device}")
    else:
        device = args.device or config.get(
            "device") or "(dry-run, brak urządzenia)"
        print("Tryb --dry-run: nie łączę się z ADB, tylko wypisuję kroki.")

    os.makedirs(args.screenshot_dir, exist_ok=True)
    if not args.dry_run:
        clear_previous_quiz_images(args.screenshot_dir)
        print("Usunięto zrzuty poprzedniego quizu; nowe pozostaną po zakończeniu.")

    ctx = {
        "device": device,
        "config": config,
        "code": args.code or config.get("code"),
        "screenshot_dir": args.screenshot_dir,
        "dry_run": args.dry_run,
        "last_screenshot": None,
        "last_quiz_result": None,
    }

    run_steps(ctx, steps)
    print("\nGotowe — scenariusz wykonany.")


if __name__ == "__main__":
    main()
