"""Execute with the calling .venv Python, save outputs and audit required content."""
import ast
import asyncio
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "CS4287-Assign2_Atari_ID1_ID2_ID3.ipynb"


def execute():
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    os.environ["MPLCONFIGDIR"] = str(ROOT / ".matplotlib")
    print("Loading notebook", flush=True)
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    # Compile all code cells before paying the cost of starting a kernel.
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"cell_{index}", "exec")
    print("All code cells compiled", flush=True)
    # Use a private, local kernelspec rather than scanning Windows/user kernels.
    # This also guarantees the kernel uses the Python that invoked this script.
    kernel_directory = ROOT / ".jupyter" / "kernels"
    spec_directory = kernel_directory / "python3"
    spec_directory.mkdir(parents=True, exist_ok=True)
    spec = {"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}",
                     "--IPKernelApp.log_level=ERROR"],
            "display_name": "PongProject verification", "language": "python"}
    (spec_directory / "kernel.json").write_text(json.dumps(spec), encoding="utf-8")
    spec_manager = KernelSpecManager(kernel_dirs=[str(kernel_directory)], ensure_native_kernel=False)
    kernel = AsyncKernelManager(kernel_name="python3", kernel_spec_manager=spec_manager)
    print("Starting local kernel with " + sys.executable, flush=True)
    client = NotebookClient(notebook, km=kernel, timeout=600, startup_timeout=60,
                            resources={"metadata": {"path": str(ROOT)}},
                            record_timing=True, allow_errors=False,
                            on_cell_executed=lambda cell, cell_index, execute_reply:
                                print(f"Executed code cell at notebook index {cell_index}", flush=True))
    started = time.perf_counter()
    try:
        client.execute(cleanup_kc=True)
    except BaseException:
        nbformat.write(notebook, NOTEBOOK)
        (ROOT / "results/notebook_execution_failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    findings = json.loads((ROOT / "results/findings.json").read_text(encoding="utf-8"))
    date = datetime.now(timezone.utc).date().isoformat()
    first = notebook.cells[0].source.splitlines()
    first[1] = (f"# Execution: ran to the end without error on {date}; "
                f"main training {findings['completed_main_runs']}/{findings['planned_runs']} complete; "
                f"shared evaluated budget {findings['common_evaluated_horizon']:,}/{findings['planned_steps_per_run']:,} decisions per run.")
    notebook.cells[0].source = "\n".join(first)
    errors = [out for cell in notebook.cells if cell.cell_type == "code"
              for out in cell.outputs if out.output_type == "error"]
    outputs = [out for cell in notebook.cells if cell.cell_type == "code" for out in cell.outputs]
    plots = sum("image/png" in out.get("data", {}) for out in outputs)
    videos = sum("<video" in str(out.get("data", {}).get("text/html", "")) for out in outputs)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    comment_lines = sum(line.lstrip().startswith("#") for c in code_cells for line in c.source.splitlines())
    unexecuted = sum(cell.execution_count is None for cell in code_cells)
    # Save real outputs before auditing, even if a presentation check needs fixing.
    nbformat.write(notebook, NOTEBOOK)
    print(json.dumps({"error_outputs": len(errors), "unexecuted": unexecuted,
                      "png_outputs": plots, "video_outputs": videos,
                      "comment_lines": comment_lines}), flush=True)
    assert not errors and unexecuted == 0, (len(errors), unexecuted)
    assert plots >= 4 and videos == 2, (plots, videos)
    assert comment_lines >= 60
    assert all(line.startswith("#") for line in first[:3])
    nbformat.validate(notebook)
    nbformat.write(notebook, NOTEBOOK)
    audit = {"executed_utc": datetime.now(timezone.utc).isoformat(), "python": sys.executable,
             "elapsed_s": time.perf_counter()-started, "code_cells": len(code_cells),
             "unexecuted_code_cells": unexecuted, "error_outputs": len(errors),
             "visible_output_blocks": len(outputs), "embedded_png_outputs": plots,
             "embedded_gameplay_videos": videos, "comment_lines": comment_lines,
             "main_runs_complete": findings["completed_main_runs"],
             "main_runs_planned": findings["planned_runs"],
             "shared_evaluated_decisions": findings["common_evaluated_horizon"],
             "team_details": "pending", "earlier_AI_use": "must be supplied by user"}
    (ROOT / "results/notebook_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    execute()
