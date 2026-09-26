import hashlib
import importlib.util
from pathlib import Path

import nbformat


def test_notebook_clean_filter_preserves_source_and_removes_outputs():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("clean_notebook", root / "scripts/clean_notebook.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("print('hello')")])
    notebook.cells[0].outputs = [nbformat.v4.new_output("stream", name="stdout", text="hello\n")]
    notebook.cells[0].execution_count = 1
    cleaned = nbformat.reads(module.clean(nbformat.writes(notebook)), as_version=4)
    assert cleaned.cells[0].source == notebook.cells[0].source
    assert cleaned.cells[0].outputs == []
    assert cleaned.cells[0].execution_count is None
    assert notebook.cells[0].outputs  # Original object is unchanged.


def test_export_extracts_exact_source_and_defaults_to_no_execution(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("build_kaggle", root / "scripts/build_kaggle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = module.build(tmp_path / "export")
    notebook = nbformat.read(target, as_version=4)
    nbformat.validate(notebook)
    namespace = {}
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        compile(cell.source, "kaggle-export", "exec")
        exec(cell.source, namespace)
        if "PROJECT_DIR" in namespace and "SOURCES" not in namespace:
            namespace["PROJECT_DIR"] = tmp_path / "extracted"
            namespace["WORK_DIR"] = tmp_path / "runs"
            namespace["INPUT_ROOT"] = tmp_path / "no-data"
    assert not namespace["selected"]
    assert not namespace["INSTALL_DEPENDENCIES"]
    for name, expected in namespace["MANIFEST"]["sha256_utf8"].items():
        data = (namespace["PROJECT_DIR"] / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected
        assert data.decode("utf-8") == (root / name).read_text(encoding="utf-8-sig")


def test_export_routes_submission_without_training_data(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("build_kaggle", root / "scripts/build_kaggle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    notebook = nbformat.read(module.build(tmp_path / "export"), as_version=4)
    codes = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    namespace = {}
    exec(codes[0], namespace)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"stub; subprocess is mocked")
    commands = []
    class Runner:
        @staticmethod
        def run(args, **kwargs):
            commands.append(args)
    namespace.update(RUN_SUBMISSION=True, CHECKPOINTS=[str(checkpoint)],
                     TEST_DIR=tmp_path, PROJECT_DIR=tmp_path,
                     WORK_DIR=tmp_path / "runs", subprocess=Runner)
    exec(next(code for code in codes if "def run_stage" in code), namespace)
    assert len(commands) == 1
    assert commands[0][2] == "submit"
    assert "--data-dir" not in commands[0]
    assert commands[0][-1] == str(checkpoint)
