"""Desktop game library for Windows and Linux. Requires Python 3.10+ and Tk."""

from __future__ import annotations

import base64
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk


WINDOWS = sys.platform == "win32"
SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent
TITLE_EXE = "game_scarlett_master.exe"
EXE_TYPES = [("Windows executables", ("*.exe", "*.EXE"))]

OUTPUT_PRESETS: tuple[tuple[str, str], ...] = (
    ("Default (Title)", ""),
    ("720p (1280x720)", "1280x720"),
    ("936p (1664x936)", "1664x936"),
    ("1080p (1920x1080)", "1920x1080"),
    ("1440p (2560x1440)", "2560x1440"),
    ("2160p (3840x2160)", "3840x2160"),
)
OUTPUT_MAP: dict[str, str] = dict(OUTPUT_PRESETS)

BG = "#101218"
PANEL = "#191c26"
FIELD = "#141620"
LINE = "#343746"
FG = "#f5f3f7"
MUTED = "#b1b4c5"
ACCENT = "#ff9b8b"
INK = "#22151b"
FONT = "Segoe UI" if WINDOWS else "TkDefaultFont"


@dataclass(frozen=True)
class Launch:
    game: Path
    log: Path
    console: str
    build: str
    fps: int
    output: str
    safe: bool
    ray_tracing: bool
    native_rt: bool
    low_end: bool
    rebuild: bool
    overrides: dict[str, str]
    runtime: Path | None = None


def data_root() -> Path:
    if WINDOWS:
        return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData/Local") / "VIctory-Unified"
    data_home = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share")
    return (data_home if data_home.is_absolute() else Path.home() / ".local/share") / "victory-unified"


@contextmanager
def library_lock(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "launcher.lock").open("a+b") as lock:
        try:
            if WINDOWS:
                import msvcrt

                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Another VIctory Unified launcher is using this game library.") from error
        yield


def save_library(path: Path, games: list[Path]) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            json.dump([str(game) for game in games], file, ensure_ascii=False, indent=2)
            file.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def load_library(path: Path) -> list[Path]:
    if path.exists():
        games = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(games, list) or not all(isinstance(game, str) for game in games):
            raise ValueError(f"Invalid game library file: {path}")
        return [Path(game) for game in games]
    candidates = ((folder / TITLE_EXE).resolve() for folder in (REPO.parent, REPO, Path.cwd()))
    games = list(dict.fromkeys(game for game in candidates if game.is_file()))
    if games:
        save_library(path, games)
    return games


def game_name(game: Path) -> str:
    return "Grand Theft Auto VI" if game.name.lower() == TITLE_EXE else game.stem


def log_path(folder: Path, game: Path, requested: str) -> Path:
    if requested:
        return (folder / Path(requested).expanduser()).resolve()
    name = re.sub(r"[^a-z0-9]+", "-", game_name(game).lower()).strip("-")[:50] or "victory"
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    return folder / f"{name}-{stamp}-{os.getpid()}.log"


def parse_overrides(text: str) -> dict[str, str]:
    overrides = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        key, equals, value = line.partition("=")
        key = key.strip().upper()
        if not equals or not re.fullmatch(r"VICTORY_[A-Z0-9_]+", key):
            raise ValueError("Use VICTORY_NAME=value for each environment override.")
        overrides[key] = value
    return overrides


def run_windows(session: Launch) -> int:
    # Send the options as data. A title path can contain spaces and apostrophes.
    payload = {
        "Exe": str(session.game),
        "Log": str(session.log),
        "Console": session.console,
        "Build": session.build,
        "Fps": session.fps,
        "Output": session.output,
        "Safe": session.safe,
        "RayTracing": session.ray_tracing,
        "NativeRT": session.native_rt,
        "Rebuild": session.rebuild,
        "More": ({**({"VICTORY_LOW_END": "1"} if session.low_end else {}), **session.overrides}),
    }
    data = base64.b64encode(json.dumps(payload).encode()).decode()
    command = """
$ErrorActionPreference = 'Stop'
$job = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('PAYLOAD')) | ConvertFrom-Json
$params = @{}
foreach ($item in $job.PSObject.Properties) { $params[$item.Name] = $item.Value }
$params.More = @{}
foreach ($item in $job.More.PSObject.Properties) { $params.More[$item.Name] = $item.Value }
try {
    & './scripts/play-rdna.ps1' @params
    exit $LASTEXITCODE
} catch {
    Add-Content -LiteralPath $job.Log -Value ($_ | Out-String) -Encoding Unicode
    exit 1
}
""".replace("PAYLOAD", data)
    encoded = base64.b64encode(command.encode("utf-16le")).decode()
    powershell = Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    return subprocess.run(
        [str(powershell), "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-EncodedCommand", encoded],
        cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW, check=False,
    ).returncode


def linux_runtime(session: Launch) -> Path:
    if session.runtime:
        candidates = [session.runtime]
    else:
        candidates = [
            REPO / "target" / session.build / "victory.exe",
            REPO / "target/x86_64-pc-windows-msvc" / session.build / "victory.exe",
        ]
    for binary in candidates:
        if binary.is_file():
            return binary
    raise FileNotFoundError(
        f"No Windows {session.build} build found. Build victory.exe on Windows and copy it here, "
        "or select it under Advanced."
    )


def run_linux(session: Launch) -> int:
    runtime = linux_runtime(session)
    wine = shutil.which("wine")
    winepath = shutil.which("winepath")
    if not wine or not winepath:
        raise FileNotFoundError("Install Wine with both wine and winepath on PATH.")

    preset = json.loads((SCRIPTS / "rdna-preset.json").read_text(encoding="utf-8"))
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("VICTORY_") or k in preset["keep"]}
    env.update(preset["base"])
    env.update(preset["safe" if session.safe else "fast"])
    if not session.safe:
        for key in (
            "VICTORY_JOIN_BARRIERS",
            "VICTORY_PORT_CBV",
            "VICTORY_PORT_GUARD",
            "VICTORY_PORT_JOURNAL",
            "VICTORY_HEAP_WORKER",
        ):
            env.setdefault(key, "1")
    if session.ray_tracing or session.native_rt:
        env["VICTORY_BVH_BUILDER"] = "1"
    if session.native_rt:
        env["VICTORY_NATIVE_RT"] = "1"
        env["VICTORY_NATIVE_RT_SUPPRESS_BUILDER"] = "1"
    if session.output:
        env["VICTORY_OUTPUT"] = session.output
    if session.fps:
        env["VICTORY_FRAME_INTERVAL_MS"] = str(round(1000 / session.fps, 3))
    if session.low_end:
        env["VICTORY_LOW_END"] = "1"
    env.update(session.overrides)

    with session.log.open("wb") as log:
        mode = "SAFE" if session.safe else "RDNA FAST"
        rt_tag = " | Native DXR 1.1" if session.native_rt else (" | BVH RT" if session.ray_tracing else "")
        low_end_tag = " | Low-End" if session.low_end else ""
        out_line = f"  Output : {session.output}\n" if session.output else ""
        header = (
            f"Launching VIctory Unified through Wine [{mode} | {session.console} | {session.build}{rt_tag}{low_end_tag}]\n"
            f"  Binary : {runtime}\n  Game   : {session.game}\n{out_line}  Log    : {session.log}\n"
        )
        log.write(header.encode("utf-8"))
        log.flush()

        def windows_path(path: str | Path) -> str:
            result = subprocess.run(
                [winepath, "-w", str(path)], cwd=session.game.parent, env=env,
                stdout=subprocess.PIPE, stderr=log, timeout=60, check=True,
            )
            converted = os.fsdecode(result.stdout.rstrip(b"\r\n"))
            if not converted:
                raise RuntimeError(f"Wine could not map this path: {path}")
            return converted

        for key in preset["keep"]:
            value = env.get(key, "")
            if value.startswith("/"):
                env[key] = windows_path(value)
        executable = windows_path(session.game)
        code = subprocess.run(
            [wine, str(runtime), "run", "--console", session.console, "--execute", executable],
            cwd=session.game.parent, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            check=False,
        ).returncode
        log.write(f"VIctory Unified exited with code {code}.\n".encode())
        return code


def run_session(session: Launch, results: queue.Queue) -> None:
    try:
        code = run_windows(session) if WINDOWS else run_linux(session)
        if code and session.log.stat().st_size == 0:
            raise RuntimeError(f"The launch process exited with code {code} before writing a log.")
        results.put((code, ""))
    except (OSError, KeyError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        message = str(error)
        try:
            with session.log.open("rb") as file:
                unicode_log = file.read(2) == b"\xff\xfe"
            with session.log.open("a", encoding="utf-16" if unicode_log else "utf-8") as file:
                file.write(f"{error}\n")
        except OSError as log_error:
            message = f"{error}\nCould not update the log: {log_error}"
        results.put((1, message))


def label(parent, text="", size=11, color=FG, bold=False, **options):
    return tk.Label(
        parent, text=text, bg=parent.cget("bg"), fg=color, anchor="w",
        font=(FONT, size, "bold" if bold else "normal"), **options,
    )


def button(parent, text, command, primary=False):
    return tk.Button(
        parent, text=text, command=command, bg=ACCENT if primary else "#272a38",
        fg=INK if primary else FG, activebackground="#ffb2a6" if primary else "#383545",
        activeforeground=INK if primary else FG, disabledforeground="#676878",
        borderwidth=0, highlightthickness=1, highlightbackground=LINE, highlightcolor=ACCENT,
        padx=13, pady=7, cursor="hand2", font=(FONT, 10, "bold" if primary else "normal"),
    )


def entry(parent, variable, width=25):
    return tk.Entry(
        parent, textvariable=variable, width=width, bg=FIELD, fg=FG, insertbackground=FG,
        relief="flat", highlightthickness=1, highlightbackground=LINE, highlightcolor=ACCENT,
    )


class Launcher:
    def __init__(self, root: tk.Tk, folder: Path):
        self.root = root
        self.library = folder / "games.json"
        self.logs = folder / "logs"
        self.games = load_library(self.library)
        self.results = queue.Queue()
        self.busy = False
        self.last_log: Path | None = None
        self.console = tk.StringVar(value="series-x")
        self.build = tk.StringVar(value="Release")
        self.fps = tk.StringVar(value="0")
        self.output = tk.StringVar(value=OUTPUT_PRESETS[0][0])
        self.safe = tk.BooleanVar()
        self.ray_tracing = tk.BooleanVar()
        self.native_rt = tk.BooleanVar()
        self.low_end = tk.BooleanVar()
        self.rebuild = tk.BooleanVar()
        self.requested_log = tk.StringVar()
        self.runtime = tk.StringVar()
        self.advanced_open = False
        self.make_window()
        self.refresh_games(0)
        self.status("Session logs are saved locally.")

    def make_window(self):
        root = self.root
        root.title("VIctory Unified")
        width = min(990, root.winfo_screenwidth() - 70)
        height = min(740, root.winfo_screenheight() - 105)
        root.geometry(f"{width}x{height}")
        root.minsize(min(width, 770), min(height, 620))
        root.configure(bg=BG, padx=20, pady=18)
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("RDNA.TCombobox", padding=4, arrowsize=13, arrowcolor=FG, bordercolor=LINE)
        style.map("RDNA.TCombobox", foreground=[("readonly", FG)],
                  fieldbackground=[("readonly", FIELD)], background=[("readonly", "#272a38")])
        root.option_add("*TCombobox*Listbox.background", FIELD)
        root.option_add("*TCombobox*Listbox.foreground", FG)
        root.option_add("*TCombobox*Listbox.selectBackground", "#543547")
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(1, weight=1)
        hero = tk.Frame(root, bg="#352536", padx=18, pady=12)
        hero.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 14))
        label(hero, "VIctory Unified", size=28, color=ACCENT, bold=True).pack(fill="x")
        platform = "Windows" if WINDOWS else "Linux · Wine"
        label(hero, f"Xbox Series game launcher  ·  Native DXR 1.1 + RDNA  ·  {platform}", color="#ead4db").pack(fill="x")

        library = tk.Frame(root, bg=PANEL, padx=13, pady=13, width=250, highlightthickness=1, highlightbackground=LINE)
        library.grid(row=1, column=0, sticky="ns", padx=(0, 13))
        library.pack_propagate(False)
        label(library, "Game library", size=15, bold=True).pack(fill="x", pady=(0, 8))
        button(library, "Add game…", self.add_game).pack(fill="x", pady=(0, 10))
        self.remove = button(library, "Remove from library", self.remove_game)
        self.remove.pack(side="bottom", anchor="w", pady=(8, 0))
        self.empty = label(library, "Add a game executable to get started.", color=MUTED, wraplength=210, justify="left")
        self.game_list = tk.Listbox(
            library, bg=PANEL, fg=FG, selectbackground="#543547", selectforeground=FG, activestyle="none",
            exportselection=False, relief="flat", borderwidth=0,
            highlightthickness=1, highlightbackground=PANEL, highlightcolor=ACCENT, font=(FONT, 11),
        )
        self.game_list.bind("<<ListboxSelect>>", lambda _event: self.selection_changed())
        self.game_list.bind("<Double-Button-1>", self.play_selected)
        self.game_list.bind("<Return>", self.play_selected)

        details = tk.Frame(root, bg=PANEL, padx=15, pady=12, highlightthickness=1, highlightbackground=LINE)
        details.grid(row=1, column=1, sticky="nsew")
        self.title = label(details, size=20, bold=True)
        self.title.pack(fill="x")
        self.path = label(details, color=MUTED, size=9)
        self.path.pack(fill="x", pady=(1, 8))
        self.caption_path = ""
        self.path.bind("<Configure>", lambda _event: self.fit_path())

        scroll_area = tk.Frame(details, bg=PANEL)
        scroll_area.pack(fill="both", expand=True)
        canvas = tk.Canvas(scroll_area, bg=PANEL, highlightthickness=0)
        scrollbar = ttk.Scrollbar(scroll_area, orient="vertical", command=canvas.yview)

        def scroll_changed(first, last):
            scrollbar.set(first, last)
            fits = float(first) == 0 and float(last) == 1
            if fits:
                scrollbar.pack_forget()
            elif not scrollbar.winfo_manager():
                scrollbar.pack(side="right", fill="y", before=canvas)

        canvas.configure(yscrollcommand=scroll_changed)
        canvas.pack(side="left", fill="both", expand=True)
        details = tk.Frame(canvas, bg=PANEL)
        settings_window = canvas.create_window(0, 0, anchor="nw", window=details)
        details.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(settings_window, width=event.width))

        controls = tk.Frame(details, bg=PANEL)
        controls.pack(fill="x")
        controls.grid_columnconfigure((0, 1), weight=1, uniform="settings")
        profile = tk.Frame(controls, bg=FIELD, padx=10, pady=8)
        profile.grid(row=0, columnspan=2, sticky="ew", pady=(0, 8))
        label(profile, "Console profile", bold=True).pack(fill="x", pady=(0, 4))
        for name, value in (("Xbox Series X", "series-x"), ("Xbox Series S", "series-s")):
            tk.Radiobutton(
                profile, text=name, value=value, variable=self.console, indicatoron=False, bg="#262734",
                fg=FG, selectcolor="#57394b", activebackground="#3d3646", activeforeground=FG,
                relief="flat", offrelief="flat", borderwidth=0, padx=10, pady=5, highlightcolor=ACCENT,
            ).pack(side="left", padx=(0, 8))

        pacing = tk.Frame(controls, bg=PANEL)
        pacing.grid(row=1, column=0, sticky="ew", padx=(0, 8))
        label(pacing, "FPS target", bold=True).pack(fill="x")
        entry(pacing, self.fps).pack(fill="x", ipady=4, pady=(3, 1))
        label(pacing, "0 uses the title's frame pacing", size=9, color=MUTED).pack(fill="x")
        build = tk.Frame(controls, bg=PANEL)
        build.grid(row=1, column=1, sticky="new")
        label(build, "Build profile", bold=True).pack(fill="x")
        ttk.Combobox(
            build, textvariable=self.build, values=("Release", "Diagnostics"), state="readonly", style="RDNA.TCombobox",
        ).pack(fill="x", pady=(3, 0))

        output_box = tk.Frame(controls, bg=PANEL)
        output_box.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        label(output_box, "Output resolution", bold=True).pack(fill="x")
        ttk.Combobox(
            output_box,
            textvariable=self.output,
            values=tuple(name for name, _ in OUTPUT_PRESETS),
            state="readonly",
            style="RDNA.TCombobox",
        ).pack(fill="x", pady=(3, 1))
        label(output_box, "Sets VICTORY_OUTPUT (Default leaves the title's resolution untouched)", size=9, color=MUTED).pack(fill="x")

        toggles = tk.Frame(controls, bg=PANEL)
        toggles.grid(row=3, columnspan=2, sticky="ew", pady=(7, 3))
        toggles_row1 = tk.Frame(toggles, bg=PANEL)
        toggles_row1.pack(fill="x")
        choices = [
            ("Safe mode", self.safe),
            ("Ray tracing (BVH)", self.ray_tracing),
            ("Native DXR 1.1 RT", self.native_rt),
        ]
        if WINDOWS:
            choices.append(("Rebuild before play", self.rebuild))
        for text, value in choices:
            tk.Checkbutton(
                toggles_row1, text=text, variable=value, bg=PANEL, fg=FG, selectcolor=FIELD,
                activebackground=PANEL, activeforeground=FG, highlightcolor=ACCENT,
            ).pack(side="left", padx=(0, 9))
        toggles_row2 = tk.Frame(toggles, bg=PANEL)
        toggles_row2.pack(fill="x", pady=(2, 0))
        tk.Checkbutton(
            toggles_row2, text="Low-End Mode (8–12 GB VRAM / 16 GB RAM)", variable=self.low_end,
            bg=PANEL, fg=FG, selectcolor=FIELD, activebackground=PANEL, activeforeground=FG,
            highlightcolor=ACCENT,
        ).pack(side="left", padx=(0, 9))

        logs = tk.Frame(details, bg=PANEL)
        logs.pack(fill="x", pady=(4, 4))
        label(logs, "Log file", bold=True).pack(fill="x", pady=(0, 3))
        button(logs, "Browse…", self.browse_log).pack(side="right", padx=(6, 0))
        entry(logs, self.requested_log).pack(fill="x", ipady=5)
        label(details, "Leave blank for a fresh timestamped log.", color=MUTED, size=9).pack(fill="x")

        self.advanced_button = button(details, "Advanced  +", self.toggle_advanced)
        self.advanced_button.pack(anchor="w", pady=(6, 0))
        self.advanced = tk.Frame(details, bg=PANEL)
        if not WINDOWS:
            label(self.advanced, "Windows runtime (.exe)", bold=True).pack(fill="x", pady=(4, 3))
            file_row = tk.Frame(self.advanced, bg=PANEL)
            file_row.pack(fill="x")
            button(file_row, "Browse…", self.browse_runtime).pack(side="right", padx=(6, 0))
            entry(file_row, self.runtime).pack(fill="x", ipady=5)
        label(self.advanced, "Environment overrides · one VICTORY_NAME=value per line", color=MUTED, size=9).pack(fill="x", pady=(4, 2))
        self.overrides = tk.Text(
            self.advanced, height=2, bg=FIELD, fg=FG, insertbackground=FG, relief="flat",
            highlightthickness=1, highlightbackground=LINE, highlightcolor=ACCENT, wrap="none", font="TkFixedFont",
        )
        self.overrides.pack(fill="x")

        footer = tk.Frame(root, bg=BG)
        footer.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        actions = tk.Frame(footer, bg=BG)
        actions.pack(side="right")
        self.status_label = label(footer, color=MUTED, size=10, wraplength=400, justify="left")
        self.status_label.pack(side="left", fill="x", expand=True, padx=(1, 9))
        button(actions, "Logs folder", lambda: self.open_path(self.logs)).pack(side="left", padx=(0, 6))
        self.open_log = button(actions, "Open last log", lambda: self.open_path(self.last_log))
        self.open_log.config(state="disabled")
        self.open_log.pack(side="left", padx=(0, 8))
        self.play = button(actions, "Play", self.play_selected, primary=True)
        self.play.config(width=8)
        self.play.pack(side="left")

    def status(self, text: str, error=False):
        self.status_label.configure(text=text, fg=ACCENT if error else MUTED)

    def refresh_games(self, index):
        self.game_list.delete(0, "end")
        if self.games:
            self.empty.pack_forget()
            self.game_list.pack(fill="both", expand=True)
            for game in self.games:
                self.game_list.insert("end", f"  {game_name(game)}")
            self.game_list.selection_set(index)
            self.game_list.see(index)
        else:
            self.game_list.pack_forget()
            self.empty.pack(fill="x")
        self.selection_changed()

    def selected(self):
        indexes = self.game_list.curselection()
        return (indexes[0], self.games[indexes[0]]) if indexes else (None, None)

    def selection_changed(self):
        _, game = self.selected()
        self.title.configure(text=game_name(game) if game else "Select a game")
        self.caption_path = str(game) if game else ""
        self.fit_path()
        self.remove.configure(state="normal" if game else "disabled")
        self.play.configure(state="normal" if game and not self.busy else "disabled")

    def fit_path(self):
        text = self.caption_path
        # Keep the executable visible when its full folder path is too wide.
        width = self.path.winfo_width()
        font = self.path.cget("font")
        if width > 1 and self.path.tk.call("font", "measure", font, text) > width:
            while text and self.path.tk.call("font", "measure", font, "…" + text) > width:
                text = text[1:]
            text = "…" + text
        self.path.configure(text=text)

    def add_game(self):
        picked = filedialog.askopenfilename(parent=self.root, title="Select an Xbox Series game executable", filetypes=EXE_TYPES)
        if not picked:
            return
        game = Path(picked).resolve()
        if game in self.games:
            self.refresh_games(self.games.index(game))
            self.status("This game is already in your library.")
            return
        try:
            updated = [*self.games, game]
            save_library(self.library, updated)
        except OSError as error:
            self.status(str(error), error=True)
            return
        self.games = updated
        self.refresh_games(len(self.games) - 1)
        self.status("Game saved to your library.")

    def remove_game(self):
        index, game = self.selected()
        if game is None:
            return
        updated = self.games[:index] + self.games[index + 1:]
        try:
            save_library(self.library, updated)
        except OSError as error:
            self.status(str(error), error=True)
            return
        self.games = updated
        self.refresh_games(min(index, len(updated) - 1))
        self.status("Game removed from your library.")

    def toggle_advanced(self):
        self.advanced_open = not self.advanced_open
        self.advanced_button.configure(text="Advanced  −" if self.advanced_open else "Advanced  +")
        if self.advanced_open:
            self.advanced.pack(fill="x")
        else:
            self.advanced.pack_forget()

    def browse_log(self):
        try:
            self.logs.mkdir(parents=True, exist_ok=True)
            picked = filedialog.asksaveasfilename(
                parent=self.root, title="Choose a new session log", initialdir=self.logs,
                defaultextension=".log", filetypes=[("Log files", "*.log"), ("All files", "*")], confirmoverwrite=False,
            )
            if picked:
                self.requested_log.set(picked)
        except OSError as error:
            self.status(str(error), error=True)

    def browse_runtime(self):
        picked = filedialog.askopenfilename(parent=self.root, title="Select a Windows build of victory.exe", filetypes=EXE_TYPES)
        if picked:
            self.runtime.set(picked)

    def open_path(self, path: Path | None):
        if path is None:
            return
        try:
            if path == self.logs:
                path.mkdir(parents=True, exist_ok=True)
            if WINDOWS:
                os.startfile(path)
            else:
                subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as error:
            self.status(str(error), error=True)

    def play_selected(self, _event=None):
        _, game = self.selected()
        if self.busy or game is None:
            return
        try:
            if not game.is_file():
                raise FileNotFoundError("The game executable is missing. Add it from its current location.")
            fps_text = self.fps.get().strip()
            if fps_text and (not fps_text.isdecimal() or len(fps_text) > 9):
                raise ValueError("Enter a whole FPS target of 0 or higher.")
            log = log_path(self.logs, game, self.requested_log.get().strip())
            runtime = self.runtime.get().strip()
            session = Launch(
                game=game,
                log=log,
                console=self.console.get(),
                build="release" if self.build.get() == "Release" else "diag",
                fps=int(fps_text or "0"),
                output=OUTPUT_MAP.get(self.output.get(), ""),
                safe=self.safe.get(),
                ray_tracing=self.ray_tracing.get(),
                native_rt=self.native_rt.get(),
                low_end=self.low_end.get(),
                rebuild=self.rebuild.get(),
                overrides=parse_overrides(self.overrides.get("1.0", "end")),
                runtime=Path(runtime).expanduser().resolve() if runtime else None,
            )
            log.parent.mkdir(parents=True, exist_ok=True)
            try:
                log.touch(exist_ok=False)
            except FileExistsError:
                raise ValueError("That log already exists. Choose a new name or clear the field.") from None
        except (OSError, ValueError) as error:
            self.status(str(error), error=True)
            return

        self.busy = True
        self.last_log = log
        self.open_log.configure(state="normal")
        self.requested_log.set("")
        self.selection_changed()
        self.status(f"Session started. {log.name}")
        threading.Thread(target=run_session, args=(session, self.results), daemon=True).start()
        self.root.after(250, self.poll_session)

    def poll_session(self):
        try:
            code, error = self.results.get_nowait()
        except queue.Empty:
            self.root.after(250, self.poll_session)
            return
        self.busy = False
        self.selection_changed()
        if error:
            self.status(error, error=True)
        elif code:
            self.status(f"Session exited with code {code}. Open the log for details.", error=True)
        else:
            self.status("Session ended. The log is ready.")


def main() -> int:
    try:
        root = tk.Tk()
    except tk.TclError as error:
        print(f"VIctory Unified could not open a desktop window: {error}", file=sys.stderr)
        return 1
    root.withdraw()
    try:
        folder = data_root()
        with library_lock(folder):
            Launcher(root, folder)
            root.deiconify()
            root.mainloop()
    except (OSError, ValueError, RuntimeError) as error:
        messagebox.showerror("VIctory Unified", str(error), parent=root)
        root.destroy()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
