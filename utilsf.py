import os

BASE_DIR = os.path.abspath("data")  # Restrict access to 'data' directory

def is_safe_path(path: str) -> bool:
    """
    Ensures the file path is within the allowed `BASE_DIR`.
    Prevents directory traversal attacks (e.g., `../../etc/passwd`).
    """
    abs_path = os.path.abspath(os.path.join(BASE_DIR, path))
    return abs_path.startswith(BASE_DIR)

def read_file(path: str) -> str:
    """
    Reads the content of a file safely.
    Returns `None` if the file does not exist.
    """
    file_path = os.path.join(BASE_DIR, path)

    if not is_safe_path(path) or not os.path.isfile(file_path):
        return None  # Return None for invalid or missing files

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
