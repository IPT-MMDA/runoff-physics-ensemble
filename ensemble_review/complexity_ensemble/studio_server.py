"""Custom browser workspace, HTTP API and native Gradio on one server."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api import create_app
from .server_settings import ServerSettings


class StudioAssets(StaticFiles):
    """Revalidate UI files so new HTML never silently keeps obsolete handlers."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


def create_studio(settings, *, native_ui=None, application=None, store=None):
    """Mount assets and Gradio without changing API authentication or job execution."""
    import gradio as gr

    app = create_app(settings, application=application, store=store)
    assets = Path(__file__).with_name("studio")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(assets / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/studio-assets", StudioAssets(directory=assets), name="studio-assets")
    if native_ui is None:
        with gr.Blocks(title="Runoff · Gradio") as native_ui:
            gr.Markdown("# Runoff research workspace\n\n"
                        "[Open the custom workspace](/) · [HTTP API documentation](/docs)\n\n"
                        "Launch with `--native-dataset DATASET_ID` to also enable the original "
                        "Gradio experiment interface here.")
    return gr.mount_gradio_app(app, native_ui, path="/gradio",
                               auth=("runoff", settings.token) if settings.token else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--native-dataset", help="Also mount the original Gradio UI for this dataset")
    args = parser.parse_args()
    token = os.environ.get("RUNOFF_API_TOKEN")
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not token:
        parser.error("set RUNOFF_API_TOKEN before binding to a non-loopback address")
    settings = ServerSettings.load(args.settings, token=token)
    native_ui = None
    if args.native_dataset:
        if args.native_dataset not in settings.datasets:
            parser.error("--native-dataset must name a registered dataset")
        from .app import build_app
        from .remote import RemoteExecutor
        dataset = settings.datasets[args.native_dataset]
        loopback = "[::1]" if args.host == "::1" else "127.0.0.1"
        native_ui = build_app(
            dataset.data_root, estreams_root=dataset.estreams_root,
            camels_chem_root=dataset.camels_chem_root, hyd_responses_root=dataset.hyd_responses_root,
            save_results=False, results_dir=settings.results_dir or settings.runs_dir,
            executor=RemoteExecutor(f"http://{loopback}:{args.port}", args.native_dataset, token))
    import uvicorn
    uvicorn.run(create_studio(settings, native_ui=native_ui), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
