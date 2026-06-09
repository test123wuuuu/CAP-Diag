#!/usr/bin/env python3
"""Clean absolute paths from code files for anonymous submission."""
import re
from pathlib import Path

def anonymize_file(file_path):
    """Replace absolute paths with relative paths."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content

    # Replace Windows-style absolute paths
    content = re.sub(
        r'D:/wuuuu/emnlp/experiments/idea5/',
        './data/',
        content
    )
    content = re.sub(
        r'D:/wuuuu/emnlp/experiments/idea6/',
        './data/',
        content
    )
    content = re.sub(
        r'D:/wuuuu/emnlp/experiments/idea6_r4/',
        './data/',
        content
    )

    # Replace generic pattern
    content = re.sub(
        r'D:/wuuuu/emnlp/[^\s"\']+',
        './data/',
        content
    )

    # Remove sys.path.insert lines that reference parent directories
    content = re.sub(
        r'sys\.path\.insert\(0,\s*str\(Path\(__file__\)\.parent\)\)',
        '# sys.path.insert removed for clean repo structure',
        content
    )

    if content != original:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    import sys
    # Force UTF-8 output on Windows
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    code_dir = Path(__file__).parent
    python_files = list(code_dir.rglob('*.py'))

    modified = 0
    for py_file in python_files:
        if py_file.name == 'anonymize_paths.py':
            continue
        if anonymize_file(py_file):
            print(f"[OK] Cleaned: {py_file.relative_to(code_dir)}")
            modified += 1

    print(f"\nTotal: {modified} files modified, {len(python_files)-1} files checked")

if __name__ == '__main__':
    main()
