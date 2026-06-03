#!/usr/bin/env python3
"""Capture and validate the advanced-gguf-quantizer interactive TUI in a PTY.

This is intentionally stdlib-only so it can run anywhere the quantizer binary
can run.  It records raw PTY bytes plus normalized screen snapshots for a small
set of keyboard-driven shell flows.
"""

from __future__ import annotations

import argparse
import dataclasses
import errno
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import shutil
import struct
import subprocess
import sys
import termios
import time
from typing import Callable, Iterable


KEY_UP = b"\x1b[A"
KEY_DOWN = b"\x1b[B"
KEY_ESC = b"\x1b"
KEY_ENTER = b"\r"

ARROW_LEAK_PATTERNS = (
    b"^[[A",
    b"^[[B",
    b"\x1b[A",
    b"\x1b[B",
)

ANSI_CSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
ANSI_OSC_RE = re.compile(r"\x1b\].*?(?:\x07|\x1b\\)", re.DOTALL)


class CaseFailure(Exception):
    """A single PTY validation case failed."""


def now() -> float:
    return time.monotonic()


def visible_bytes(data: bytes) -> str:
    text = data.decode("utf-8", "replace")
    text = text.replace("\x1b", "^[")
    text = text.replace("\r", "\\r")
    text = text.replace("\n", "\\n")
    return text


def normalize_transcript(raw: bytes) -> str:
    text = raw.decode("utf-8", "replace")
    text = ANSI_OSC_RE.sub("", text)
    text = ANSI_CSI_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x1b", "^[")
    return text


def set_winsize(fd: int, rows: int, cols: int) -> None:
    packed = struct.pack("HHHH", rows, cols, 0, 0)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, packed)


class TerminalScreen:
    """Small ANSI screen model for the quantizer shell's clear/redraw output."""

    def __init__(self, rows: int = 32, cols: int = 120) -> None:
        self.rows = rows
        self.cols = cols
        self.row = 0
        self.col = 0
        self.buf = [[" "] * cols for _ in range(rows)]
        self.state = "normal"
        self.csi = ""

    def feed(self, data: bytes) -> None:
        for ch in data.decode("utf-8", "replace"):
            self._feed_char(ch)

    def text(self) -> str:
        lines = ["".join(row).rstrip() for row in self.buf]
        while lines and not lines[-1]:
            lines.pop()
        return "\n".join(lines)

    def _feed_char(self, ch: str) -> None:
        if self.state == "esc":
            if ch == "[":
                self.state = "csi"
                self.csi = ""
            else:
                self.state = "normal"
            return

        if self.state == "csi":
            self.csi += ch
            if "@" <= ch <= "~":
                self._handle_csi(self.csi[:-1], ch)
                self.csi = ""
                self.state = "normal"
            return

        if ch == "\x1b":
            self.state = "esc"
        elif ch == "\r":
            self.col = 0
        elif ch == "\n":
            self._newline()
        elif ch == "\b":
            self.col = max(0, self.col - 1)
        elif ch == "\t":
            for _ in range(4 - (self.col % 4)):
                self._put(" ")
        elif ch >= " ":
            self._put(ch)

    def _handle_csi(self, params: str, final: str) -> None:
        private = params.startswith("?")
        if private:
            params = params[1:]
        values = self._parse_params(params)

        if final in ("H", "f"):
            row = values[0] if len(values) >= 1 and values[0] > 0 else 1
            col = values[1] if len(values) >= 2 and values[1] > 0 else 1
            self.row = min(self.rows - 1, max(0, row - 1))
            self.col = min(self.cols - 1, max(0, col - 1))
        elif final == "J":
            mode = values[0] if values else 0
            if mode in (2, 3):
                self._clear()
            elif mode == 0:
                self._clear_to_end()
        elif final == "K":
            self._clear_line_from_cursor()
        elif final == "A":
            self.row = max(0, self.row - (values[0] if values else 1))
        elif final == "B":
            self.row = min(self.rows - 1, self.row + (values[0] if values else 1))
        elif final == "C":
            self.col = min(self.cols - 1, self.col + (values[0] if values else 1))
        elif final == "D":
            self.col = max(0, self.col - (values[0] if values else 1))
        elif final == "G":
            col = values[0] if values else 1
            self.col = min(self.cols - 1, max(0, col - 1))

    @staticmethod
    def _parse_params(params: str) -> list[int]:
        if not params:
            return []
        out: list[int] = []
        for item in params.split(";"):
            if not item:
                out.append(0)
                continue
            try:
                out.append(int(item))
            except ValueError:
                out.append(0)
        return out

    def _put(self, ch: str) -> None:
        if self.col >= self.cols:
            self._newline()
        self.buf[self.row][self.col] = ch
        self.col += 1
        if self.col >= self.cols:
            self._newline()

    def _newline(self) -> None:
        self.row += 1
        self.col = 0
        if self.row >= self.rows:
            self.buf.pop(0)
            self.buf.append([" "] * self.cols)
            self.row = self.rows - 1

    def _clear(self) -> None:
        self.buf = [[" "] * self.cols for _ in range(self.rows)]

    def _clear_to_end(self) -> None:
        self._clear_line_from_cursor()
        for r in range(self.row + 1, self.rows):
            self.buf[r] = [" "] * self.cols

    def _clear_line_from_cursor(self) -> None:
        for c in range(self.col, self.cols):
            self.buf[self.row][c] = " "


@dataclasses.dataclass
class Snapshot:
    label: str
    seconds: float
    screen: str


@dataclasses.dataclass
class Action:
    seconds: float
    label: str
    bytes: str


class PtySession:
    def __init__(
        self,
        binary: Path,
        case_dir: Path,
        timeout: float,
        rows: int,
        cols: int,
        cwd: Path,
    ) -> None:
        self.binary = binary
        self.case_dir = case_dir
        self.timeout = timeout
        self.rows = rows
        self.cols = cols
        self.cwd = cwd
        self.screen = TerminalScreen(rows, cols)
        self.raw = bytearray()
        self.snapshots: list[Snapshot] = []
        self.actions: list[Action] = []
        self.started = now()
        self.master_fd: int | None = None
        self.proc: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        self.case_dir.mkdir(parents=True, exist_ok=True)
        (self.case_dir / "home").mkdir(exist_ok=True)
        (self.case_dir / "xdg").mkdir(exist_ok=True)

        master_fd, slave_fd = os.openpty()
        set_winsize(slave_fd, self.rows, self.cols)
        os.set_blocking(master_fd, False)

        env = os.environ.copy()
        env.update(
            {
                "TERM": "xterm-256color",
                "NO_COLOR": "1",
                "XDG_CONFIG_HOME": str(self.case_dir / "xdg"),
                "LANG": env.get("LANG", "C.UTF-8"),
                "LC_ALL": env.get("LC_ALL", "C.UTF-8"),
            }
        )

        try:
            self.proc = subprocess.Popen(
                [str(self.binary)],
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                cwd=self.cwd,
                env=env,
                close_fds=True,
                start_new_session=True,
            )
        finally:
            os.close(slave_fd)
        self.master_fd = master_fd

    def write(self, data: bytes, label: str) -> None:
        if self.proc is None or self.master_fd is None:
            raise RuntimeError("session not started")
        if self.proc.poll() is not None:
            raise CaseFailure(f"process exited before input: {label}")
        self.actions.append(Action(now() - self.started, label, visible_bytes(data)))
        os.write(self.master_fd, data)

    def line(self, text: str, label: str | None = None) -> None:
        self.write(text.encode("utf-8") + KEY_ENTER, label or f"line {text!r}")

    def key(self, key: bytes, label: str) -> None:
        self.write(key, label)

    def read_for(self, seconds: float) -> None:
        deadline = now() + seconds
        while now() < deadline:
            self._read_available()
            time.sleep(0.02)
        self._read_available()

    def wait_screen(self, needle: str, timeout: float | None = None) -> None:
        timeout = self.timeout if timeout is None else timeout

        def contains() -> bool:
            return needle in self.screen.text()

        self.wait_until(contains, timeout, f"screen contains {needle!r}")

    def wait_exit(self, timeout: float | None = None) -> int:
        timeout = self.timeout if timeout is None else timeout
        deadline = now() + timeout
        while now() < deadline:
            self._read_available()
            if self.proc is not None and self.proc.poll() is not None:
                self._read_available()
                return int(self.proc.returncode)
            time.sleep(0.02)
        raise CaseFailure(f"process did not exit within {timeout:.1f}s")

    def wait_until(self, predicate: Callable[[], bool], timeout: float, description: str) -> None:
        deadline = now() + timeout
        while now() < deadline:
            self._read_available()
            self.assert_no_arrow_leaks()
            if predicate():
                self.read_for(0.05)
                self.assert_no_arrow_leaks()
                return
            if self.proc is not None and self.proc.poll() is not None:
                self._read_available()
                self.assert_no_arrow_leaks()
                if predicate():
                    return
                raise CaseFailure(f"process exited before {description}")
            time.sleep(0.02)
        self.snapshot(f"timeout waiting for {description}")
        raise CaseFailure(f"timed out waiting for {description}")

    def snapshot(self, label: str) -> None:
        self._read_available()
        self.snapshots.append(Snapshot(label, now() - self.started, self.screen.text()))

    def assert_no_arrow_leaks(self) -> None:
        found = []
        for pattern in ARROW_LEAK_PATTERNS:
            if pattern in self.raw:
                shown = visible_bytes(pattern)
                if shown not in found:
                    found.append(shown)
        if found:
            raise CaseFailure("literal arrow escape leaked into PTY output: " + ", ".join(found))

    def close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.proc.wait(timeout=1.0)
        self._read_available()
        if self.master_fd is not None:
            try:
                os.close(self.master_fd)
            except OSError:
                pass
            self.master_fd = None

    def write_artifacts(self, status: str, error: str | None = None) -> None:
        self.case_dir.mkdir(parents=True, exist_ok=True)
        (self.case_dir / "transcript.raw").write_bytes(bytes(self.raw))
        (self.case_dir / "transcript.normalized.txt").write_text(
            normalize_transcript(bytes(self.raw)),
            encoding="utf-8",
        )
        (self.case_dir / "snapshots.txt").write_text(self._snapshots_text(), encoding="utf-8")
        (self.case_dir / "snapshots.json").write_text(
            json.dumps([dataclasses.asdict(s) for s in self.snapshots], indent=2) + "\n",
            encoding="utf-8",
        )
        (self.case_dir / "actions.json").write_text(
            json.dumps([dataclasses.asdict(a) for a in self.actions], indent=2) + "\n",
            encoding="utf-8",
        )
        result = {
            "status": status,
            "error": error,
            "returncode": None if self.proc is None else self.proc.poll(),
            "seconds": now() - self.started,
        }
        (self.case_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    def _snapshots_text(self) -> str:
        parts: list[str] = []
        for snap in self.snapshots:
            parts.append(f"=== {snap.label} ({snap.seconds:.3f}s) ===")
            parts.append(snap.screen)
            parts.append("")
        return "\n".join(parts)

    def _read_available(self) -> None:
        if self.master_fd is None:
            return
        while True:
            try:
                chunk = os.read(self.master_fd, 65536)
            except BlockingIOError:
                return
            except OSError as exc:
                if exc.errno in (errno.EIO, errno.EBADF):
                    return
                raise
            if not chunk:
                return
            self.raw.extend(chunk)
            self.screen.feed(chunk)


def enter_slash_command(session: PtySession, rest: str, label: str | None = None) -> None:
    session.key(b"/", "slash command trigger")
    session.wait_screen("New value >")
    session.line(rest, label or f"slash command /{rest}")


def quit_with_slash(session: PtySession) -> None:
    enter_slash_command(session, "quit")
    rc = session.wait_exit()
    if rc != 0:
        raise CaseFailure(f"/quit exited with status {rc}")


def back_to_main_and_quit(session: PtySession) -> None:
    session.key(KEY_ESC, "Esc Back to main menu")
    session.wait_screen("Main Menu")
    quit_with_slash(session)


def assert_screen_contains(session: PtySession, needle: str) -> None:
    if needle not in session.screen.text():
        session.snapshot(f"missing {needle!r}")
        raise CaseFailure(f"screen did not contain {needle!r}")


def assert_screen_not_contains(session: PtySession, needle: str) -> None:
    if needle in session.screen.text():
        session.snapshot(f"unexpected {needle!r}")
        raise CaseFailure(f"screen unexpectedly contained {needle!r}")


def create_project_to_project_menu(session: PtySession, name: str) -> None:
    session.wait_screen("Main Menu")
    session.key(KEY_ENTER, "Enter create project")
    session.wait_screen("Project Name")
    session.line(name, "type project name")

    session.wait_screen("Create New Project > Save Location")
    assert_screen_contains(session, "Use This Directory")
    session.key(KEY_ENTER, "Use This Directory")

    session.wait_screen("Project > Options > Quant Type")
    session.key(KEY_ENTER, "Select NVFP4")

    session.wait_screen("Create New Project > Next")
    session.key(KEY_DOWN, "Down to Open project")
    session.wait_screen(">  2.  Open project")
    session.key(KEY_ENTER, "Open project")

    session.wait_screen("Pick BF16/input GGUF")
    session.snapshot("created project menu")


def write_minimal_project(case_dir: Path) -> Path:
    project = case_dir / "load-validation.bwqproj"
    project.write_text(
        '{"schema":"advanced-gguf-quantizer-project-v1",'
        '"event":"project_init",'
        '"created_at":"2026-05-22T00:00:00-0700",'
        '"name":"load validation",'
        '"recipe":"",'
        '"input":"",'
        '"baseline":{"bf16_reference":"","kld_base":"","corpus":"","imatrix":""},'
        '"notes":""}\n',
        encoding="utf-8",
    )
    return project


def write_fixture_files(case_dir: Path) -> dict[str, Path]:
    fixtures = case_dir / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    files = {
        "gguf": fixtures / "tiny.gguf",
        "eval": fixtures / "eval.txt",
        "calibration": fixtures / "calibration.txt",
        "kld": fixtures / "base.kld",
        "imatrix": fixtures / "imatrix.dat",
        "tensor_file": fixtures / "tensor-types.txt",
        "recipe": fixtures / "recipe-with-ngl.toml",
        "output": fixtures / "tiny-NVFP4.gguf",
        "run_dir": fixtures / "run",
        "candidates": fixtures / "candidates",
    }
    files["gguf"].write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 0))
    files["eval"].write_text("The quick brown fox tests KLD setup.\n", encoding="utf-8")
    files["calibration"].write_text("Calibration text for imatrix setup.\n", encoding="utf-8")
    files["kld"].write_bytes(b"not-a-real-kld-but-selectable\n")
    files["imatrix"].write_bytes(b"not-a-real-imatrix-but-selectable\n")
    files["tensor_file"].write_text("blk.0.attn_q.weight=Q8_0\n", encoding="utf-8")
    files["recipe"].write_text(
        "\n".join(
            [
                "version = 1",
                "",
                "[io]",
                f'input = "{files["gguf"]}"',
                f'output = "{files["output"]}"',
                "",
                "[calibration]",
                f'corpus = "{files["calibration"]}"',
                f'imatrix = "{files["imatrix"]}"',
                'n_gpu_layers = "auto"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return files


def select_down(session: PtySession, count: int, label: str) -> None:
    for _ in range(count):
        session.key(KEY_DOWN, label)
        session.read_for(0.02)
    session.key(KEY_ENTER, label)


def continue_if_paused(session: PtySession, next_screen: str) -> None:
    session.wait_screen("Press Enter to continue.")
    session.key(KEY_ENTER, "continue")
    if next_screen == "Project > Options":
        wait_options_root(session)
    else:
        session.wait_screen(next_screen)


def wait_options_root(session: PtySession) -> None:
    def at_root() -> bool:
        screen = session.screen.text()
        return (
            "Project > Options" in screen
            and "Select primary quant type" in screen
            and "Native candidate search" in screen
        )

    session.wait_until(at_root, session.timeout, "Project options root")


def picker_select_typed_path(session: PtySession, path: Path, label: str) -> None:
    session.wait_screen("Type Path")
    screen = session.screen.text()
    selected_match = re.search(r"^>\s+(\d+)\.", screen, re.MULTILINE)
    type_match = re.search(r"^\s*>?\s+(\d+)\.\s+Type Path\b", screen, re.MULTILINE)
    if type_match is None:
        raise CaseFailure("picker did not offer Type Path")
    selected = int(selected_match.group(1)) if selected_match else 1
    target = int(type_match.group(1))
    for _ in range(max(0, target - selected)):
        session.key(KEY_DOWN, label + " move to Type Path")
        session.read_for(0.02)
    for _ in range(max(0, selected - target)):
        session.key(KEY_UP, label + " move to Type Path")
        session.read_for(0.02)
    session.key(KEY_ENTER, label + " open Type Path")
    session.wait_screen("Path")
    session.line(str(path), label)


def assert_contextual_prompt(session: PtySession, label: str, title: str | None = None) -> None:
    session.wait_screen(label)
    screen = session.screen.text()
    has_context = (
        "advanced-gguf-quantizer" in screen
        or "Project >" in screen
        or "Command >" in screen
        or "| Status" in screen
        or "Session" in screen
    )
    if not has_context:
        session.snapshot("missing prompt context")
        raise CaseFailure("prompt screen did not include page or session context")
    if title is not None:
        assert_screen_contains(session, title)


def press_prompt_defaults(session: PtySession, labels: Iterable[str]) -> None:
    for label in labels:
        assert_contextual_prompt(session, label)
        session.key(KEY_ENTER, f"default {label}")


def configure_model_input_from_project(session: PtySession, gguf: Path) -> None:
    session.wait_screen("Pick BF16/input GGUF")
    session.key(KEY_ENTER, "Enter model files")
    session.wait_screen("Project > Model Files")
    select_down(session, 1, "Type model input path")
    assert_contextual_prompt(session, "Model Input", "Project > Model Files")
    session.line(str(gguf), "model input path")
    session.wait_screen("Pick BF16/input GGUF")


def open_options_from_project(session: PtySession) -> None:
    session.wait_screen("Pick BF16/input GGUF")
    select_down(session, 2, "Open options")
    session.wait_screen("Project > Options")


def open_options_item(session: PtySession, index: int, label: str) -> None:
    wait_options_root(session)
    select_down(session, index, label)


def run_quality_choice(session: PtySession, index: int, files: dict[str, Path], name: str) -> None:
    open_options_item(session, 2, "Quality inputs")
    session.wait_screen("Project > Options > Quality Inputs")
    select_down(session, index, name)
    if index == 0:
        picker_select_typed_path(session, files["eval"], "existing KLD corpus")
        picker_select_typed_path(session, files["kld"], "existing KLD base")
    elif index == 1:
        picker_select_typed_path(session, files["eval"], "make KLD corpus")
        press_prompt_defaults(session, ["KLD base output", "perplexity executable"])
    elif index == 2:
        picker_select_typed_path(session, files["calibration"], "calibration corpus")
        picker_select_typed_path(session, files["imatrix"], "imatrix file")
    elif index == 3:
        picker_select_typed_path(session, files["calibration"], "make imatrix corpus")
        press_prompt_defaults(
            session,
            [
                "imatrix output",
                "llama-imatrix executable",
                "CPU threads",
                "CPU batch/collector threads",
                "imatrix context size",
                "imatrix batch size",
                "imatrix ubatch size",
            ],
        )
        assert_contextual_prompt(session, "imatrix GPU layers", "Make Imatrix")
        session.line("auto", "imatrix GPU layers")
        press_prompt_defaults(session, ["extra llama-imatrix args"])
    elif index == 4:
        picker_select_typed_path(session, files["eval"], "bundle corpus")
        press_prompt_defaults(session, ["KLD base output", "perplexity executable", "eval bundle path"])
    continue_if_paused(session, "Project > Options")


def case_main_menu_simplicity(session: PtySession, _case_dir: Path) -> None:
    session.wait_screen("Main Menu")
    session.snapshot("main menu simplicity")

    for label in (
        "Create New Project",
        "Load Existing Project",
        "Inspect GGUF",
        "Quit",
    ):
        assert_screen_contains(session, label)

    for old_label in (
        "Create or edit recipe",
        "Show PPL/KLD best-candidate report",
        "JSONL",
        "best-candidate report",
    ):
        assert_screen_not_contains(session, old_label)

    quit_with_slash(session)


def case_main_menu_navigation(session: PtySession, _case_dir: Path) -> None:
    session.wait_screen("Main Menu")
    session.snapshot("main menu initial")

    session.key(KEY_DOWN, "Down")
    session.wait_screen(">  2.  Load Existing Project")
    session.snapshot("main menu after Down")

    session.key(KEY_UP, "Up")
    session.wait_screen(">  1.  Create New Project")
    session.snapshot("main menu after Up")

    session.key(b"j", "j navigation")
    session.wait_screen(">  2.  Load Existing Project")
    session.snapshot("main menu after j")

    session.key(b"k", "k navigation")
    session.wait_screen(">  1.  Create New Project")
    session.snapshot("main menu after k")

    quit_with_slash(session)


def case_bare_esc_exits_main_menu(session: PtySession, _case_dir: Path) -> None:
    session.wait_screen("Main Menu")
    session.snapshot("main menu before bare Esc")
    session.key(KEY_ESC, "bare Esc")
    rc = session.wait_exit(timeout=2.0)
    if rc != 0:
        raise CaseFailure(f"bare Esc exited with status {rc}")


def case_slash_command_with_injected_arrows(session: PtySession, _case_dir: Path) -> None:
    session.wait_screen("Main Menu")
    session.snapshot("main menu before injected slash command")
    session.key(b"/", "slash command trigger")
    session.wait_screen("New value >")
    session.write(KEY_UP + b"commands" + KEY_DOWN + KEY_ENTER, "Up + text + Down in slash prompt")
    session.wait_screen("Commands")
    session.wait_screen("Press Enter to continue.")
    session.snapshot("slash command help after injected arrows")
    session.key(KEY_ENTER, "continue after slash command help")
    session.wait_screen("Main Menu")
    quit_with_slash(session)


def case_load_project_cancel_back(session: PtySession, _case_dir: Path) -> None:
    session.wait_screen("Main Menu")
    session.key(KEY_DOWN, "Down to Load existing project")
    session.wait_screen(">  2.  Load Existing Project")
    session.key(KEY_ENTER, "Enter load project")
    session.wait_screen("Load project")
    session.snapshot("load project chooser")
    session.key(KEY_ESC, "Esc cancel load project")
    session.wait_screen("Main Menu")
    session.snapshot("load project cancelled")
    quit_with_slash(session)


def case_create_project_back(session: PtySession, _case_dir: Path) -> None:
    session.wait_screen("Main Menu")
    session.snapshot("main menu before create project")
    session.key(KEY_ENTER, "Enter create project")
    session.wait_screen("Project Name")
    session.snapshot("create project name prompt")
    session.line("PTY validation project", "type project name")

    session.wait_screen("Create New Project > Save Location")
    session.snapshot("create project save-location chooser")
    session.key(KEY_ESC, "Esc Back from create project save location")
    session.wait_screen("Main Menu")
    session.snapshot("create project cancelled")
    quit_with_slash(session)


def case_create_project_name_esc_back(session: PtySession, _case_dir: Path) -> None:
    session.wait_screen("Main Menu")
    session.key(KEY_ENTER, "Enter create project")
    session.wait_screen("Project Name")
    session.snapshot("project-name prompt before arrows and Esc")
    session.write(KEY_UP + KEY_DOWN + KEY_ESC, "Up + Down + bare Esc in project-name prompt")
    session.wait_screen("Main Menu")
    session.snapshot("create project cancelled from name prompt")
    quit_with_slash(session)


def case_created_project_hides_start_until_ready(session: PtySession, _case_dir: Path) -> None:
    create_project_to_project_menu(session, "No Start Validation")
    assert_screen_contains(session, "Project")
    assert_screen_contains(session, "Pick BF16/input GGUF")
    assert_screen_contains(session, "Choose options")
    assert_screen_contains(session, "Review and start pipeline")
    back_to_main_and_quit(session)


def case_mxfp6_notice_visible(session: PtySession, _case_dir: Path) -> None:
    session.wait_screen("Main Menu")
    session.key(KEY_ENTER, "Enter create project")
    session.wait_screen("Project Name")
    session.line("MXFP6 Notice Validation", "type project name")

    session.wait_screen("Create New Project > Save Location")
    session.key(KEY_ENTER, "Use This Directory")

    session.wait_screen("Project > Options > Quant Type")
    session.key(KEY_DOWN, "Down to MXFP6")
    session.wait_screen(">  2.  MXFP6")
    session.key(KEY_ENTER, "Select MXFP6")

    session.wait_screen("format may change")
    assert_screen_contains(session, "experimental")
    assert_screen_contains(session, "unsupported")
    assert_screen_contains(session, "feedback")
    assert_screen_contains(session, "mxfp6-cuda")
    session.snapshot("mxfp6 notice after quant type selection")
    session.key(KEY_ESC, "Esc to project menu")
    session.wait_screen("Project")
    back_to_main_and_quit(session)


def case_create_project_colon_name_uses_fallback(session: PtySession, _case_dir: Path) -> None:
    create_project_to_project_menu(session, ":")
    assert_screen_contains(session, "advanced-gguf-quantizer")
    assert_screen_contains(session, "Pick BF16/input GGUF")
    assert_screen_contains(session, "Review and start pipeline")
    back_to_main_and_quit(session)


def case_load_project_manual_path_opens_menu(session: PtySession, case_dir: Path) -> None:
    project = write_minimal_project(case_dir)
    session.wait_screen("Main Menu")
    session.key(KEY_DOWN, "Down to Load existing project")
    session.wait_screen(">  2.  Load Existing Project")
    session.key(KEY_ENTER, "Enter load project")
    session.wait_screen("Load project")
    session.wait_screen("Type Path")
    session.key(KEY_ENTER, "Enter manual project path")
    session.wait_screen("Path")
    session.write(KEY_UP + str(project).encode("utf-8") + KEY_DOWN + KEY_ENTER, "arrows + manual project path")
    session.wait_screen("Pick BF16/input GGUF")
    session.snapshot("loaded project menu")
    assert_screen_not_contains(session, "Start Quantization")
    session.key(KEY_ESC, "Esc Back from project menu")
    session.wait_screen("Main Menu")
    quit_with_slash(session)


def case_nested_quit_exits_from_project_menu(session: PtySession, case_dir: Path) -> None:
    project = write_minimal_project(case_dir)
    session.wait_screen("Main Menu")
    session.key(KEY_DOWN, "Down to Load existing project")
    session.wait_screen(">  2.  Load Existing Project")
    session.key(KEY_ENTER, "Enter load project")
    session.wait_screen("Load project")
    session.wait_screen("Type Path")
    session.key(KEY_ENTER, "Enter manual project path")
    session.wait_screen("Path")
    session.line(str(project), "manual project path")
    session.wait_screen("Pick BF16/input GGUF")
    session.snapshot("project menu before nested quit")
    quit_with_slash(session)


def case_kld_required_flow(session: PtySession, _case_dir: Path) -> None:
    create_project_to_project_menu(session, "KLD Required Flow")

    session.key(KEY_DOWN, "Down to quant type")
    session.key(KEY_DOWN, "Down to options")
    session.wait_screen("Choose options")
    session.key(KEY_ENTER, "Enter options")
    session.wait_screen("Project > Options")
    assert_screen_contains(session, "Model input and output")

    session.key(KEY_DOWN, "Down to Model input/output")
    session.key(KEY_DOWN, "Down to Quality inputs")
    session.wait_screen("Quality inputs")
    session.key(KEY_ENTER, "Enter quality inputs menu")
    session.wait_screen("Project > Options > Quality Inputs")

    assert_screen_contains(session, "Use existing KLD base")
    assert_screen_contains(session, "Make KLD base from BF16")
    assert_screen_contains(session, "Select calibration corpus and imatrix")
    assert_screen_not_contains(session, "Continue without KLD")
    session.snapshot("KLD required menu")
    session.key(KEY_ESC, "Esc Back from KLD menu")

    session.wait_screen("Project > Options")
    session.key(KEY_ESC, "Esc Back from options")
    session.wait_screen("Pick BF16/input GGUF")
    session.snapshot("project menu after KLD required flow")
    back_to_main_and_quit(session)


def case_options_esc_returns_without_pause(session: PtySession, case_dir: Path) -> None:
    project = write_minimal_project(case_dir)
    session.wait_screen("Main Menu")
    session.key(KEY_DOWN, "Down to Load existing project")
    session.wait_screen(">  2.  Load Existing Project")
    session.key(KEY_ENTER, "Enter load project")
    session.wait_screen("Load project")
    session.wait_screen("Type Path")
    session.key(KEY_ENTER, "Enter manual project path")
    session.wait_screen("Path")
    session.line(str(project), "manual project path")
    session.wait_screen("Pick BF16/input GGUF")

    session.key(KEY_DOWN, "Down to quant type")
    session.key(KEY_DOWN, "Down to options")
    session.wait_screen("Choose options")
    session.key(KEY_ENTER, "Enter options")
    session.wait_screen("Project > Options")
    session.snapshot("options before Esc")
    session.key(KEY_ESC, "Esc Back from options")
    session.wait_screen("Pick BF16/input GGUF")
    assert_screen_not_contains(session, "Press Enter to continue.")
    session.snapshot("project menu after options Esc")
    back_to_main_and_quit(session)


def case_load_recipe_accepts_calibration_ngl(session: PtySession, case_dir: Path) -> None:
    files = write_fixture_files(case_dir)
    session.wait_screen("Main Menu")
    enter_slash_command(session, f"config {files['recipe']}")
    session.wait_screen("loaded")
    assert_screen_not_contains(session, "unknown recipe field")
    session.wait_screen("Press Enter to continue.")
    session.key(KEY_ENTER, "continue after config load")
    session.wait_screen("Main Menu")
    saved = files["recipe"].read_text(encoding="utf-8")
    if 'n_gpu_layers = "auto"' not in saved:
        raise CaseFailure("loaded recipe did not preserve calibration.n_gpu_layers")
    quit_with_slash(session)


def case_quality_inputs_all_paths(session: PtySession, case_dir: Path) -> None:
    files = write_fixture_files(case_dir)
    create_project_to_project_menu(session, "Quality Inputs Exhaustive")
    configure_model_input_from_project(session, files["gguf"])
    open_options_from_project(session)

    run_quality_choice(session, 0, files, "Use existing KLD base")
    run_quality_choice(session, 1, files, "Make KLD base")
    run_quality_choice(session, 2, files, "Select calibration corpus")
    run_quality_choice(session, 3, files, "Make imatrix")
    run_quality_choice(session, 4, files, "Eval bundle")

    open_options_item(session, 2, "Quality inputs back")
    session.wait_screen("Project > Options > Quality Inputs")
    session.key(KEY_ESC, "Back from quality inputs")
    session.wait_screen("Project > Options")
    session.key(KEY_ESC, "Back from options")
    session.wait_screen("Pick BF16/input GGUF")
    back_to_main_and_quit(session)


def case_nvfp4_options_prompt_pages(session: PtySession, case_dir: Path) -> None:
    files = write_fixture_files(case_dir)
    create_project_to_project_menu(session, "NVFP4 Options Exhaustive")
    configure_model_input_from_project(session, files["gguf"])
    open_options_from_project(session)

    open_options_item(session, 3, "Target BPW / VRAM")
    session.wait_screen("Target BPW / VRAM > Fit")
    select_down(session, 5, "Type custom VRAM")
    assert_contextual_prompt(session, "Custom VRAM target", "Target BPW / VRAM")
    session.line("24", "custom VRAM")
    press_prompt_defaults(session, ["Target final average BPW", "KV/cache reserve GiB", "activation/headroom reserve GiB"])
    continue_if_paused(session, "Project > Options")

    open_options_item(session, 4, "NVFP4 4/6 Policy")
    press_prompt_defaults(session, ["NVFP4 preset", "raw NVFP4 cfg override"])
    session.wait_screen("Lane Selection")
    session.key(KEY_ENTER, "Adaptive lane selection")
    assert_contextual_prompt(session, "4/6 refit iterations", "Parameters")
    press_prompt_defaults(
        session,
        [
            "4/6 refit iterations",
            "4/6 companding enabled",
            "6-bit lane cap",
            "4-bit lane cap",
            "NVFP4 scale correction denominator",
            "input scale policy",
            "max sample blocks",
            "CPU worker threads",
            "calibration/search families",
            "scale/group tie policy",
        ],
    )
    continue_if_paused(session, "Project > Options")

    open_options_item(session, 6, "PPL/KLD scoring gates")
    press_prompt_defaults(
        session,
        [
            "mean KLD penalty",
            "p99 KLD penalty",
            "p999 KLD penalty",
            "max KLD penalty",
            "mean KLD max delta",
            "p99 KLD max delta",
            "p999 KLD max delta",
            "max KLD max delta",
            "hard gate mean KLD",
            "hard gate p99 KLD",
            "hard gate p999 KLD",
            "hard gate max KLD",
        ],
    )
    continue_if_paused(session, "Project > Options")

    open_options_item(session, 7, "Edit Existing GGUF")
    press_prompt_defaults(
        session,
        [
            "enable Edit Existing GGUF pass",
            "edit quant type",
            "tensors to edit",
            "report top tensors",
            "edit budget MiB",
            "BF16 edit budget MiB",
            "per-class edit limit",
            "NVFP4 retest top",
            "edit sample blocks",
            "edit coarse max blocks",
            "edit refine max blocks",
            "edit guard max blocks",
            "edit report file",
            "edit tensor type overrides",
        ],
    )
    continue_if_paused(session, "Project > Options")

    open_options_item(session, 8, "Tensor rules")
    session.wait_screen("Project > Options > Tensor Rules")
    session.key(KEY_ENTER, "Browse model tensors")
    session.wait_screen("Project > Options > Tensor Rules")
    select_down(session, 1, "Add tensor override")
    assert_contextual_prompt(session, "tensor=type", "Add Override")
    session.line("blk.0.attn_q.weight=Q8_0", "tensor override")
    session.wait_screen("Project > Options > Tensor Rules")
    select_down(session, 2, "Add tensor override file")
    assert_contextual_prompt(session, "tensor override file", "Add Override File")
    session.line(str(files["tensor_file"]), "tensor override file")
    session.wait_screen("Project > Options > Tensor Rules")
    select_down(session, 3, "Set edit tensor list")
    assert_contextual_prompt(session, "edit tensor types", "Tensor Rules > Edit Existing GGUF")
    session.line("blk.0.ffn_gate.weight=Q6_K", "edit tensor types")
    session.wait_screen("Project > Options > Tensor Rules")
    session.key(KEY_ESC, "Back from tensor rules")
    continue_if_paused(session, "Project > Options")

    open_options_item(session, 9, "Standard quantize options")
    session.wait_screen("Project > Options > Standard Quantize Options")
    select_down(session, 1, "Select standard options")
    press_prompt_defaults(
        session,
        [
            "output.weight tensor type",
            "token embedding tensor type",
            "MTP/NextN tensor type",
            "leave output.weight unquantized",
            "measure embeddings/output as candidates",
            "token embedding candidate types",
            "output tensor candidate types",
            "minimum tensor savings MiB",
            "technique candidates",
            "allow requantizing already-quantized tensors",
            "pure mode",
            "copy only",
            "keep input split/shard layout",
            "prune layers CSV",
            "add metadata override",
            "add tensor type override file",
            "add tensor override tensor=type",
        ],
    )
    continue_if_paused(session, "Project > Options")

    open_options_item(session, 10, "Save config")
    press_prompt_defaults(session, ["save configuration as", "run directory"])
    continue_if_paused(session, "Project > Options")

    session.key(KEY_ESC, "Back from options")
    session.wait_screen("Pick BF16/input GGUF")
    back_to_main_and_quit(session)


def case_candidate_search_all_paths(session: PtySession, case_dir: Path) -> None:
    files = write_fixture_files(case_dir)
    create_project_to_project_menu(session, "Candidate Search Exhaustive")
    configure_model_input_from_project(session, files["gguf"])
    open_options_from_project(session)

    open_options_item(session, 5, "Native candidate search")
    session.wait_screen("Candidate Search")
    session.key(KEY_ENTER, "Select native technique families")
    session.wait_screen("Native Technique Families")
    session.key(KEY_ENTER, "Review current native families")
    session.wait_screen("Current Native Search")
    session.key(KEY_ENTER, "continue native review")
    session.wait_screen("Native Technique Families")
    for index, label in enumerate(
        [
            "Use core PPL/KLD search",
            "Use full local GGUF search set",
            "Auto-search candidate seed",
            "Tensor-policy sweep",
            "Add BF16/no-quantize tensor choices",
            "AWQ candidates",
            "SmoothQuant input-scale candidates",
            "Scale/cap sweep candidates",
            "NVFP4 RSF variants",
            "KL-divergence sensitivity scorer",
            "Gradient/Hessian sidecar scorer",
            "Tie grouped decisions",
        ],
        start=1,
    ):
        select_down(session, index, label)
        session.wait_screen("Native Technique Families")
    select_down(session, 13, "Manual native lists")
    press_prompt_defaults(session, ["technique candidates", "calibration/search families", "scale/group tie policy"])
    session.wait_screen("Native Technique Families")
    session.key(KEY_ESC, "Back from native techniques")
    continue_if_paused(session, "Project > Options")
    open_options_item(session, 5, "Native candidate search")

    for index, label in [(1, "Keep current search settings"), (2, "Fast"), (3, "Normal"), (4, "Deep")]:
        session.wait_screen("Candidate Search")
        select_down(session, index, label)
        if index == 1:
            session.wait_screen("Project > Options")
        else:
            continue_if_paused(session, "Project > Options")
        open_options_item(session, 5, label)

    session.wait_screen("Candidate Search")
    select_down(session, 5, "Advanced low-level knobs")
    press_prompt_defaults(
        session,
        [
            "KLD base file",
            "selector evidence ledger",
            "candidate search checkpoint GGUF",
            "checkpoint cache directory",
            "skip remaining tuning request file",
            "effort label",
            "stage A sample blocks",
            "stage A max policies",
            "refine top policies",
            "refine budget",
            "survey top tensors",
            "survey sample blocks",
            "max tensors to search",
            "full PPL/KLD candidates",
            "eval sequences",
            "policy search threads",
            "selector threads",
            "PPL/KLD host reduction threads",
            "keep generated checkpoint",
            "require resident tensor cache",
            "write selector trace",
            "selector only",
        ],
    )
    continue_if_paused(session, "Project > Options")
    session.key(KEY_ESC, "Back from options")
    session.wait_screen("Pick BF16/input GGUF")
    back_to_main_and_quit(session)


def case_mxfp6_options_visible_and_usable(session: PtySession, case_dir: Path) -> None:
    files = write_fixture_files(case_dir)
    session.wait_screen("Main Menu")
    session.key(KEY_ENTER, "Enter create project")
    session.wait_screen("Project Name")
    session.line("MXFP6 Options Exhaustive", "type project name")
    session.wait_screen("Create New Project > Save Location")
    session.key(KEY_ENTER, "Use This Directory")
    session.wait_screen("Project > Options > Quant Type")
    session.key(KEY_DOWN, "Down to MXFP6")
    session.wait_screen(">  2.  MXFP6")
    session.key(KEY_ENTER, "Select MXFP6")
    session.wait_screen("Create New Project > Next")
    session.key(KEY_DOWN, "Down to Open project")
    session.key(KEY_ENTER, "Open project")
    session.wait_screen("Pick BF16/input GGUF")
    configure_model_input_from_project(session, files["gguf"])
    open_options_from_project(session)
    open_options_item(session, 7, "MXFP6 scale refinement")
    press_prompt_defaults(
        session,
        [
            "MXFP6 tensor scale mode",
            "MXFP6 min savings bytes",
            "scale-refine tensor count",
            "scale candidates",
        ],
    )
    continue_if_paused(session, "Project > Options")
    session.key(KEY_ESC, "Back from options")
    session.wait_screen("Pick BF16/input GGUF")
    back_to_main_and_quit(session)


def case_project_review_eval_inspect(session: PtySession, case_dir: Path) -> None:
    files = write_fixture_files(case_dir)
    create_project_to_project_menu(session, "Project Review Eval Inspect")
    configure_model_input_from_project(session, files["gguf"])

    select_down(session, 4, "Review and start pipeline")
    session.wait_screen("Project > Review and Start")
    session.key(KEY_ENTER, "Show blockers")
    session.wait_screen("Resolve these before quantization")
    session.key(KEY_ENTER, "continue blockers")
    session.wait_screen("Project > Review and Start")
    session.key(KEY_ESC, "Back from review start")
    session.wait_screen("Press Enter to continue.")
    session.key(KEY_ENTER, "continue after review start")
    session.wait_screen("Pick BF16/input GGUF")

    select_down(session, 5, "Show status")
    session.wait_screen("Project Status")
    session.key(KEY_ENTER, "continue status")
    session.wait_screen("Pick BF16/input GGUF")

    select_down(session, 6, "Evaluation and best candidates")
    session.wait_screen("Project > Evaluation and Best Candidates")
    session.key(KEY_ENTER, "Record metrics")
    press_prompt_defaults(session, ["project path", "variant id"])
    for label, value in [
        ("PPL", "1.0"),
        ("mean KLD", "0.01"),
        ("p99 KLD", "0.02"),
        ("p999 KLD", "0.03"),
        ("max KLD", "0.04"),
        ("BPW", "4.5"),
    ]:
        assert_contextual_prompt(session, label)
        session.line(value, label)
    session.wait_screen("Project > Evaluation and Best Candidates")
    select_down(session, 1, "Run best-candidate report")
    press_prompt_defaults(session, ["project path", "metrics export path"])
    session.wait_screen("Project > Evaluation and Best Candidates")
    select_down(session, 2, "Generate candidate configs")
    assert_contextual_prompt(session, "candidate output directory", "Generate Candidates")
    session.line(str(files["candidates"]), "candidate output directory")
    session.wait_screen("Project > Evaluation and Best Candidates")
    session.key(KEY_ESC, "Back from evaluation")
    session.wait_screen("Pick BF16/input GGUF")

    select_down(session, 7, "Inspect GGUF")
    session.wait_screen("GGUF")
    session.key(KEY_ENTER, "continue inspect")
    session.wait_screen("Pick BF16/input GGUF")
    back_to_main_and_quit(session)


CASES: dict[str, Callable[[PtySession, Path], None]] = {
    "main_menu_simplicity": case_main_menu_simplicity,
    "main_menu_navigation": case_main_menu_navigation,
    "bare_esc_exits_main_menu": case_bare_esc_exits_main_menu,
    "slash_command_with_injected_arrows": case_slash_command_with_injected_arrows,
    "load_project_cancel_back": case_load_project_cancel_back,
    "create_project_back": case_create_project_back,
    "create_project_name_esc_back": case_create_project_name_esc_back,
    "created_project_hides_start_until_ready": case_created_project_hides_start_until_ready,
    "mxfp6_notice_visible": case_mxfp6_notice_visible,
    "create_project_colon_name_uses_fallback": case_create_project_colon_name_uses_fallback,
    "load_project_manual_path_opens_menu": case_load_project_manual_path_opens_menu,
    "nested_quit_exits_from_project_menu": case_nested_quit_exits_from_project_menu,
    "kld_required_flow": case_kld_required_flow,
    "options_esc_returns_without_pause": case_options_esc_returns_without_pause,
    "load_recipe_accepts_calibration_ngl": case_load_recipe_accepts_calibration_ngl,
    "quality_inputs_all_paths": case_quality_inputs_all_paths,
    "nvfp4_options_prompt_pages": case_nvfp4_options_prompt_pages,
    "candidate_search_all_paths": case_candidate_search_all_paths,
    "mxfp6_options_visible_and_usable": case_mxfp6_options_visible_and_usable,
    "project_review_eval_inspect": case_project_review_eval_inspect,
}


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PTY capture/validation cases against an advanced-gguf-quantizer TUI binary.",
    )
    parser.add_argument("binary", type=Path, help="path to advanced-gguf-quantizer")
    parser.add_argument("--out", type=Path, required=True, help="directory for transcripts and snapshots")
    parser.add_argument("--case", choices=sorted(CASES), action="append", help="case to run; may be repeated")
    parser.add_argument("--timeout", type=float, default=20.0, help="default per-wait timeout in seconds")
    parser.add_argument("--rows", type=int, default=34, help="PTY rows")
    parser.add_argument("--cols", type=int, default=120, help="PTY columns")
    parser.add_argument("--cwd", type=Path, default=Path.cwd(), help="working directory for the TUI")
    return parser.parse_args(list(argv))


def resolve_binary(binary: Path) -> Path:
    if binary.is_absolute():
        resolved = binary
    else:
        resolved = Path.cwd() / binary
    if not resolved.exists():
        raise SystemExit(f"advanced-gguf-quantizer binary not found: {resolved}")
    if not os.access(resolved, os.X_OK):
        raise SystemExit(f"advanced-gguf-quantizer binary is not executable: {resolved}")
    return resolved


def run_one(
    name: str,
    fn: Callable[[PtySession, Path], None],
    binary: Path,
    out_dir: Path,
    timeout: float,
    rows: int,
    cols: int,
    cwd: Path,
) -> dict[str, object]:
    case_dir = out_dir / name
    if case_dir.exists():
        shutil.rmtree(case_dir)
    session = PtySession(binary, case_dir, timeout, rows, cols, cwd)
    error: str | None = None
    status = "pass"
    try:
        session.start()
        fn(session, case_dir)
        session.assert_no_arrow_leaks()
    except Exception as exc:
        status = "fail"
        error = str(exc)
        if isinstance(exc, CaseFailure):
            pass
        else:
            error = f"{type(exc).__name__}: {exc}"
    finally:
        session.close()
        session.write_artifacts(status, error)

    return {
        "case": name,
        "status": status,
        "error": error,
        "dir": str(case_dir),
    }


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    binary = resolve_binary(args.binary)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = args.case or list(CASES)
    results = [
        run_one(
            name,
            CASES[name],
            binary,
            out_dir,
            args.timeout,
            args.rows,
            args.cols,
            args.cwd,
        )
        for name in selected
    ]

    summary = {
        "binary": str(binary),
        "out": str(out_dir),
        "results": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    failed = [result for result in results if result["status"] != "pass"]
    for result in results:
        prefix = "PASS" if result["status"] == "pass" else "FAIL"
        line = f"{prefix} {result['case']} -> {result['dir']}"
        if result["error"]:
            line += f": {result['error']}"
        print(line)
    print(f"summary: {out_dir / 'summary.json'}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
