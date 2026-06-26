from __future__ import annotations

import shutil
import subprocess


def choose_directory(title: str = "Select project folder") -> str | None:
    if shutil.which("zenity"):
        completed = subprocess.run(
            ["zenity", "--file-selection", "--directory", f"--title={title}"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            return completed.stdout.strip() or None
        if completed.returncode == 1:
            return None
        raise RuntimeError(completed.stderr.strip() or "Zenity folder picker failed")

    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Tkinter is not available in this Python installation") from exc

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        selected = filedialog.askdirectory(title=title, mustexist=True)
    finally:
        root.destroy()
    return selected or None
