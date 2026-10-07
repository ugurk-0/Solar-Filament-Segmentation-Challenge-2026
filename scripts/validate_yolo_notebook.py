"""Validate the YOLO presentation cells in an existing, selected Jupyter kernel.

Does not start training or interrupt the kernel. Saves static outputs only; the
live widget is recreated by running its notebook cell.
"""
import argparse
from pathlib import Path
import time
import nbformat
from jupyter_client import BlockingKernelClient


def execute(client, code, timeout=60):
    message_id = client.execute(code, store_history=False)
    outputs = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        message = client.get_iopub_msg(timeout=max(1, deadline-time.monotonic()))
        if message.get('parent_header', {}).get('msg_id') != message_id:
            continue
        kind, content = message['msg_type'], message['content']
        if kind == 'error':
            raise RuntimeError('\n'.join(content['traceback']))
        if kind in ('display_data', 'execute_result', 'stream'):
            outputs.append(nbformat.v4.output_from_msg(message))
        if kind == 'status' and content['execution_state'] == 'idle':
            return outputs
    raise TimeoutError('Kernel did not finish; no interrupt sent')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('connection_file')
    args = parser.parse_args()
    client = BlockingKernelClient(connection_file=args.connection_file)
    client.load_connection_file()
    client.start_channels()
    try:
        client.wait_for_ready(timeout=15)
        probe = execute(client, 'print(__import__("sys").executable); print(__import__("os").getcwd())')
        for output in probe:
            if output.output_type == 'stream':
                print(output.text.strip())
        notebook = nbformat.read('notebook.ipynb', as_version=4)
        static = next(c for c in notebook.cells if 'yolo-dashboard-1' in c.metadata.get('tags', []))
        live = next(c for c in notebook.cells if 'yolo-dashboard-2' in c.metadata.get('tags', []))
        static.outputs = execute(client, static.source)
        static.execution_count = None
        assert sum('image/png' in o.get('data', {}) for o in static.outputs) == 3
        execute(client, live.source)
        execute(client, 'assert YOLO_PANEL.task is not None and not YOLO_PANEL.task.done(); YOLO_PANEL.refresh()')
        # Release this validation panel. The notebook cell starts its own panel.
        execute(client, 'YOLO_PANEL.close()')
        live.outputs = []
        live.execution_count = None
        nbformat.validate(notebook)
        nbformat.write(notebook, 'notebook.ipynb')
        print('Validated: selected kernel imports, static table/curve/two preview images, live widget and refresh callback.')
    finally:
        client.stop_channels()


if __name__ == '__main__':
    main()
