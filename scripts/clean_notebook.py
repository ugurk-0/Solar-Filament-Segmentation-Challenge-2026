"""Git clean filter: strip notebook outputs from the staged copy only."""
import sys

import nbformat


def clean(text):
    notebook = nbformat.reads(text, as_version=4)
    notebook.metadata.pop("widgets", None)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
        cell.metadata.pop("execution", None)
    return nbformat.writes(notebook) + "\n"


if __name__ == "__main__":
    sys.stdout.buffer.write(clean(sys.stdin.buffer.read().decode("utf-8-sig")).encode("utf-8"))
