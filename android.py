#!/usr/bin/env python3
"""Odczytuje współrzędne dotknięć telefonu przez ADB.

Użycie: python3 android.py --record-taps
"""

import argparse
import os
import random
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET

COMMON_ADB_LOCATIONS = [
    # macOS (Android Studio)
    os.path.expanduser("~/Library/Android/sdk/platform-tools/adb"),
    # Linux (Android Studio)
    os.path.expanduser("~/Android/Sdk/platform-tools/adb"),
    os.path.expanduser(
        "~/AppData/Local/Android/Sdk/platform-tools/adb.exe"),  # Windows
    "/usr/local/bin/adb",
    "/opt/homebrew/bin/adb",
]

_ADB_BIN: str | None = None
DEVICE_DUMP_PATH = "/sdcard/window_dump.xml"
LOCAL_DUMP_PATH = "/tmp/window_dump.xml"


def resolve_adb_binary(explicit_path: str | None = None) -> str:
    """Znajduje binarkę adb."""
    global _ADB_BIN
    if _ADB_BIN:
        return _ADB_BIN

    candidates = []
    if explicit_path:
        candidates.append(explicit_path)

    for env_var in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
        sdk_dir = os.environ.get(env_var)
        if sdk_dir:
            candidates.append(os.path.join(sdk_dir, "platform-tools", "adb"))

    which_result = shutil.which("adb")
    if which_result:
        candidates.append(which_result)

    candidates.extend(COMMON_ADB_LOCATIONS)

    for path in candidates:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            _ADB_BIN = path
            return path

    raise RuntimeError(
        "Nie mogę znaleźć binarki 'adb'. Sprawdziłem PATH, zmienne "
        "ANDROID_HOME/ANDROID_SDK_ROOT oraz typowe ścieżki instalacji "
        "Android Studio (np. ~/Library/Android/sdk/platform-tools/adb na macOS).\n"
        "Podaj jawnie ścieżkę przez: --adb-path ~/Library/Android/sdk/platform-tools/adb\n"
        "albo dodaj platform-tools do PATH w swoim ~/.zshrc:\n"
        '  export PATH="$HOME/Library/Android/sdk/platform-tools:$PATH"'
    )


def adb_base_cmd(device: str | None):
    cmd = [resolve_adb_binary()]
    if device:
        cmd += ["-s", device]
    return cmd


def run_adb(device: str | None, args, timeout: int = 15) -> subprocess.CompletedProcess:
    result = subprocess.run(adb_base_cmd(device) + args, capture_output=True,
                            text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "Polecenie adb nie powiodło się.")
    return result


def adb_connect(address: str):
    """Łączy się przez ADB po Wi-Fi (adb connect IP:PORT)."""
    result = subprocess.run(
        [resolve_adb_binary(), "connect", address], capture_output=True, text=True, timeout=15
    )
    print(
        f"[adb connect {address}] {result.stdout.strip() or result.stderr.strip()}")
    if "connected" not in result.stdout.lower() and "already" not in result.stdout.lower():
        raise RuntimeError(
            f"Nie udało się połączyć z {address}: {result.stdout} {result.stderr}")


def list_devices():
    result = subprocess.run(
        [resolve_adb_binary(), "devices"], capture_output=True, text=True, timeout=10
    )
    lines = [ln.strip() for ln in result.stdout.splitlines()[1:] if ln.strip()]
    devices = [ln.split("\t")[0]
               for ln in lines if "device" in ln and "offline" not in ln]
    return devices


def ensure_device_available(device: str | None):
    devices = list_devices()
    if not devices:
        raise RuntimeError(
            "Nie widzę żadnego podłączonego/połączonego urządzenia ADB. "
            "Sprawdź USB debugging / `adb connect IP:PORT` i uruchom `adb devices`."
        )
    if device and device not in devices:
        raise RuntimeError(
            f"Urządzenie '{device}' nie jest na liście dostępnych: {devices}"
        )
    if not device and len(devices) > 1:
        print(
            f"[uwaga] Widzę kilka urządzeń: {devices}. Użyję pierwszego: {devices[0]}")
    return device or devices[0]


def dump_ui_xml(device: str) -> str:
    """Zrzuca bieżącą hierarchię UI urządzenia jako XML."""
    run_adb(device, ["shell", "uiautomator", "dump", DEVICE_DUMP_PATH])
    run_adb(device, ["pull", DEVICE_DUMP_PATH, LOCAL_DUMP_PATH])
    with open(LOCAL_DUMP_PATH, "r", encoding="utf-8") as f:
        return f.read()


def center_of(bounds_str: str):
    """Zwraca środek granic Androida w formacie ``[x1,y1][x2,y2]``."""
    match = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds_str)
    if not match:
        return None
    x1, y1, x2, y2 = (int(value) for value in match.groups())
    return (x1 + x2) // 2, (y1 + y2) // 2


def find_nodes_by_text(xml_str: str, needle: str):
    """Zwraca węzły UI, których tekst lub opis zawiera ``needle``."""
    needle_norm = needle.strip().lower()
    matches = []
    for node in ET.fromstring(xml_str).iter("node"):
        text = (node.get("text") or "").strip()
        desc = (node.get("content-desc") or "").strip()
        if needle_norm in f"{text} {desc}".lower():
            matches.append(node.attrib)
    return matches


def find_input_field(xml_str: str, hint_words=("kod", "code")):
    """Znajduje widoczne pole EditText, preferując pole z podpowiedzią kodu."""
    fields = [node.attrib for node in ET.fromstring(xml_str).iter("node")
              if "EditText" in (node.get("class") or "") and node.get("bounds")]
    for field in fields:
        content = " ".join(field.get(key, "") for key in
                           ("text", "content-desc", "resource-id")).lower()
        if any(word in content for word in hint_words):
            return field
    return fields[0] if fields else None


def find_webview_node(xml_str: str):
    """Zwraca pierwszy węzeł WebView albo ``None``."""
    for node in ET.fromstring(xml_str).iter("node"):
        if "WebView" in (node.get("class") or ""):
            return node.attrib
    return None


def tap(device: str, x: int, y: int):
    run_adb(device, ["shell", "input", "tap", str(x), str(y)])


def type_text(device: str, text: str):
    run_adb(device, ["shell", "input", "text", text])


def press_enter(device: str):
    run_adb(device, ["shell", "input", "keyevent", "66"])


def take_screenshot(device: str, local_path: str):
    """Wykonuje zrzut ekranu telefonu i zapisuje go jako PNG."""
    result = subprocess.run(adb_base_cmd(device) + ["exec-out", "screencap", "-p"],
                            capture_output=True, timeout=20)
    if result.returncode or not result.stdout.startswith(b"\x89PNG"):
        raise RuntimeError("Nie udało się wykonać poprawnego zrzutu ekranu PNG.")
    with open(local_path, "wb") as f:
        f.write(result.stdout)


def wait_for_and_click_text(device: str, search_text: str, timeout: float,
                            poll_interval: float):
    """Czeka na element zawierający tekst i naciska środek jego granic."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        for node in find_nodes_by_text(dump_ui_xml(device), search_text):
            xy = center_of(node.get("bounds", ""))
            if xy:
                print(f"Klikam {search_text!r} w ({xy[0]}, {xy[1]})")
                tap(device, *xy)
                return True
        time.sleep(poll_interval)
    raise TimeoutError(f"Nie znalazłem elementu zawierającego tekst: {search_text!r}.")


def wait_for_input_field_and_type(device: str, code: str, timeout: float,
                                  poll_interval: float, hint_words=("kod", "code")):
    """Czeka na pole tekstowe, ustawia fokus i wpisuje wartość."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        field = find_input_field(dump_ui_xml(device), hint_words)
        xy = center_of(field.get("bounds", "")) if field else None
        if xy:
            tap(device, *xy)
            time.sleep(0.5)
            type_text(device, code)
            return True
        time.sleep(poll_interval)
    raise TimeoutError("Nie znalazłem pola do wpisania kodu.")


def wait_for_webview(device: str, timeout: float, poll_interval: float):
    """Czeka na pojawienie się WebView w hierarchii UI."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        xml_str = dump_ui_xml(device)
        if find_webview_node(xml_str):
            return xml_str
        time.sleep(poll_interval)
    raise TimeoutError(f"Karta WebView nie pojawiła się w ciągu {timeout}s.")


def tap_checkbox_fixed(device: str, x: int, y: int, delay_min: float, delay_max: float):
    """Naciska checkbox pod znanymi współrzędnymi po krótkim opóźnieniu."""
    time.sleep(random.uniform(delay_min, delay_max))
    tap(device, x, y)


# Kody zdarzeń evdev Multi-Touch typu B.
EV_ABS = "0003"
ABS_MT_TRACKING_ID = "0039"
ABS_MT_POSITION_X = "0035"
ABS_MT_POSITION_Y = "0036"
TRACKING_ID_RELEASE = 0xffffffff

RAW_EVENT_LINE_RE = re.compile(
    r"^(?:/dev/input/(?:event\d+):\s+)?([0-9a-fA-F]{4})\s+([0-9a-fA-F]{4})\s+([0-9a-fA-F]{8})$"
)


def run_adb_capture_until_idle(device: str, args, idle_timeout: float = 1.5,
                               max_total: float = 8.0) -> str:
    """Czyta output ADB, aż ucichnie, a potem kończy proces."""
    import select
    cmd = adb_base_cmd(device) + args
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            text=True, bufsize=1)
    collected = []
    start = time.time()
    try:
        while True:
            remaining = max_total - (time.time() - start)
            if remaining <= 0:
                break
            ready, _, _ = select.select(
                [proc.stdout], [], [], min(idle_timeout, remaining))
            if not ready:
                break  # cisza przez idle_timeout -> koniec statycznego outputu
            line = proc.stdout.readline()
            if not line:
                break
            collected.append(line)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
    return "".join(collected)


def list_touch_event_devices(device: str):
    """
    Zwraca listę ścieżek /dev/input/eventX, które zgłaszają ABS_MT_POSITION_X
    (czyli wyglądają na ekran dotykowy), na podstawie `adb shell getevent -pl`.
    """
    output = run_adb_capture_until_idle(device, ["shell", "getevent", "-pl"])
    devices = []
    current_path = None
    has_touch = False
    for line in output.splitlines():
        m = re.match(r"add device \d+: (/dev/input/event\d+)", line.strip())
        if m:
            if current_path and has_touch:
                devices.append(current_path)
            current_path = m.group(1)
            has_touch = False
            continue
        if current_path and "ABS_MT_POSITION_X" in line:
            has_touch = True
    if current_path and has_touch:
        devices.append(current_path)
    return devices


def guess_touch_event_device(device: str) -> str:
    """Zgaduje ścieżkę ekranu dotykowego; rzuca błąd z podpowiedzią, jeśli się nie da."""
    devices = list_touch_event_devices(device)
    if not devices:
        raise RuntimeError(
            "Nie znalazłem żadnego /dev/input/eventX z ABS_MT_POSITION_X (ekran "
            "dotykowy). Podaj ścieżkę ręcznie przez --event-device — sprawdź ją "
            "poleceniem `adb shell getevent -pl` (szukaj urządzenia z "
            "ABS_MT_POSITION_X / ABS_MT_POSITION_Y w opisie)."
        )
    if len(devices) > 1:
        print(f"[uwaga] Kilku kandydatów na ekran dotykowy: {devices}. "
              f"Używam pierwszego: {devices[0]}. Jeśli to zła ścieżka, wskaż "
              f"właściwą przez --event-device.")
    return devices[0]


def get_abs_mt_range(device: str, event_path: str):
    """
    Zwraca ((min_x, max_x), (min_y, max_y)) — surowy zakres wartości
    ABS_MT_POSITION_X/Y raportowany przez sterownik dotyku. Zwykle NIE jest
    to to samo co rozdzielczość ekranu w pikselach (np. dotyk raportuje
    0..4095, a ekran ma 1080x2400), dlatego trzeba przeskalować.
    """
    output = run_adb_capture_until_idle(
        device, ["shell", "getevent", "-pl", event_path])

    def find_range(axis_name):
        m = re.search(axis_name + r".*?min (-?\d+), max (-?\d+)", output)
        return (int(m.group(1)), int(m.group(2))) if m else None

    return find_range("ABS_MT_POSITION_X"), find_range("ABS_MT_POSITION_Y")


def get_screen_size(device: str):
    """Zwraca (width, height) rozdzielczości ekranu z `adb shell wm size`."""
    result = run_adb(device, ["shell", "wm", "size"], timeout=10)
    m = re.search(r"Physical size:\s*(\d+)x(\d+)", result.stdout) or \
        re.search(r"Override size:\s*(\d+)x(\d+)", result.stdout)
    if not m:
        raise RuntimeError(
            f"Nie udało się odczytać rozdzielczości ekranu z: {result.stdout!r}")
    return int(m.group(1)), int(m.group(2))


def record_taps(device: str, event_path: str | None, idle_timeout: float = 60.0):
    """Wypisuje współrzędne dotknięć do Ctrl+C lub po minucie bezczynności."""
    if not event_path:
        event_path = guess_touch_event_device(device)
    print(f"Nasłuchuję zdarzeń dotyku na {event_path}...")

    try:
        x_range, y_range = get_abs_mt_range(device, event_path)
        screen_w, screen_h = get_screen_size(device)
        print(
            f"Zakres surowy: X={x_range} Y={y_range}; ekran: {screen_w}x{screen_h}")
    except Exception as exc:
        print(f"[uwaga] Nie udało się ustalić zakresu/rozdzielczości ({exc}) — "
              f"pokażę tylko surowe (raw) współrzędne, bez przeliczenia na piksele.")
        x_range = y_range = None
        screen_w = screen_h = None

    def scale(raw, raw_range, screen_max):
        if raw is None or not raw_range or screen_max is None:
            return None
        raw_min, raw_max = raw_range
        if raw_max == raw_min:
            return None
        frac = (raw - raw_min) / (raw_max - raw_min)
        return max(0, min(screen_max, int(round(frac * screen_max))))

    # exec-out ogranicza problemy z buforowaniem getevent.
    cmd = adb_base_cmd(device) + ["exec-out", "getevent", event_path]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            text=True, bufsize=1)

    last_x = last_y = None
    taps_found = 0
    lines_seen = 0

    print("Stuknij teraz w telefonie w miejsce, które chcesz sprawdzić "
          "(Ctrl+C, żeby zakończyć wcześniej).\n")
    try:
        import select
        while True:
            ready, _, _ = select.select([proc.stdout], [], [], idle_timeout)
            if not ready:
                print(
                    f"\nBrak zdarzeń przez {idle_timeout}s — kończę nasłuchiwanie.")
                if lines_seen == 0:
                    print(
                        "Nie doszła ŻADNA linia ze strumienia getevent — to nie jest "
                        "kwestia braku stuknięcia, tylko połączenie/urządzenie nic nie "
                        "wysłało. Sprawdź ręcznie, czy to w ogóle coś zwraca:\n"
                        f"  adb -s {device} exec-out getevent {event_path}\n"
                        "(uruchom to bezpośrednio w terminalu i stuknij w ekran — jeśli "
                        "tam też cisza, sprawdź czy to na pewno właściwy --event-device.)"
                    )
                break
            line = proc.stdout.readline()
            if not line:
                break
            lines_seen += 1
            m = RAW_EVENT_LINE_RE.match(line.strip())
            if not m:
                continue
            ev_type, ev_code, ev_value_hex = m.groups()
            ev_type, ev_code = ev_type.lower(), ev_code.lower()
            value = int(ev_value_hex, 16)

            if ev_type != EV_ABS:
                continue
            if ev_code == ABS_MT_POSITION_X:
                last_x = value
            elif ev_code == ABS_MT_POSITION_Y:
                last_y = value
            elif ev_code == ABS_MT_TRACKING_ID and value == TRACKING_ID_RELEASE:
                # Palec odklejony -> (last_x, last_y) to komplet współrzędnych tego dotknięcia.
                screen_x = scale(last_x, x_range, screen_w -
                                 1 if screen_w else None)
                screen_y = scale(last_y, y_range, screen_h -
                                 1 if screen_h else None)
                taps_found += 1
                print(f"[tap #{taps_found}] raw=({last_x}, {last_y}) "
                      f"-> screen=({screen_x}, {screen_y})   "
                      f"# adb shell input tap {screen_x} {screen_y}")
    except KeyboardInterrupt:
        print("\nPrzerwano (Ctrl+C).")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()

    if taps_found == 0:
        print("\nNie zarejestrowałem żadnego dotknięcia.")


def main():
    parser = argparse.ArgumentParser(
        description="Odczytuje współrzędne dotknięć przez ADB."
    )
    parser.add_argument("--device", default=None,
                        help="Numer seryjny urządzenia (adb -s ...)")
    parser.add_argument("--connect", default=None,
                        help="Adres do 'adb connect' (np. 192.168.1.50:5555)")
    parser.add_argument(
        "--adb-path",
        default=None,
        help="Jawna ścieżka do binarki adb, gdy nie jest w PATH "
             "(np. ~/Library/Android/sdk/platform-tools/adb)",
    )

    parser.add_argument("--record-taps", action="store_true",
                        help="Nasłuchuj dotknięć i wypisz ich współrzędne.")
    parser.add_argument("--event-device", default=None,
                        help="Ścieżka /dev/input/eventX (domyślnie wykrywana).")
    args = parser.parse_args()

    if not args.record_taps:
        parser.error("Podaj --record-taps.")

    if args.adb_path:
        resolve_adb_binary(explicit_path=os.path.expanduser(args.adb_path))
    else:
        resolve_adb_binary()
    print(f"Używam adb: {_ADB_BIN}")

    if args.connect:
        adb_connect(args.connect)

    device = ensure_device_available(args.device)
    print(f"Używam urządzenia: {device}")

    record_taps(device, args.event_device)


if __name__ == "__main__":
    main()
