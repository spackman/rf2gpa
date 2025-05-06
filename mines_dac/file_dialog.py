import tempfile
import subprocess
import pickle
import sys
import os
import textwrap

def select_file_exe(title: str = "Select a spectra file") -> str:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(title=title)
    root.destroy()
    return file_path

# For use in the background (runs with VS Code)
def select_file_bkg(title: str = "Select a spectra file") -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
        tmp_path = tmp_file.name

    gui_script = textwrap.dedent(f"""
        import tkinter as tk
        from tkinter import filedialog
        import pickle
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(title="{title}")
        with open(r'{tmp_path}', 'wb') as f:
            pickle.dump(file_path, f)
    """)

    result = subprocess.run([sys.executable, '-c', gui_script], capture_output=True, text=True)

    if result.returncode != 0:
        print("Error: Failed to open file dialog.")
        print(result.stderr)
        return ""

    try:
        with open(tmp_path, 'rb') as f:
            return pickle.load(f)
    finally:
        os.remove(tmp_path)
