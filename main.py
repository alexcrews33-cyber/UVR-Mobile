
import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from audio_separator.separator import Separator

app = FastAPI(title="UVR Mobile API", version="1.0.0")

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/tmp/uvr-output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.getenv(
    "UVR_MODEL",
    "UVR-MDX-NET-Inst_HQ_3.onnx"
)

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))


@app.get("/")
def home():
    return {
        "service": "UVR Mobile API",
        "status": "online",
        "engine": "Audio Separator",
        "processing": "real audio separation"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/process")
async def process_audio(
    file: UploadFile = File(...),
    options: str = Form("vocals")
):
    if not file.filename:
        raise HTTPException(400, "Missing filename")

    allowed = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}
    suffix = Path(file.filename).suffix.lower()

    if suffix not in allowed:
        raise HTTPException(400, "Unsupported audio format")

    if options not in {"vocals", "instrumental"}:
        raise HTTPException(
            400,
            "Choose vocals or instrumental"
        )

    workdir = Path(tempfile.mkdtemp(prefix="uvr-"))

    try:
        input_path = workdir / f"input{suffix}"

        size = 0
        with input_path.open("wb") as destination:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break

                size += len(chunk)
                if size > MAX_UPLOAD_MB * 1024 * 1024:
                    raise HTTPException(
                        413,
                        f"File exceeds {MAX_UPLOAD_MB} MB"
                    )

                destination.write(chunk)

        if size == 0:
            raise HTTPException(400, "Audio file is empty")

        separator = Separator(
            output_dir=str(workdir),
            output_format="WAV"
        )

        separator.load_model(model_filename=MODEL_NAME)
        output_files = separator.separate(str(input_path))

        if not output_files:
            raise RuntimeError("The model produced no output")

        # UVR models commonly produce two stems.
        # Select the requested stem using the output filename.
        keywords = (
            ("vocal", "voice")
            if options == "vocals"
            else ("instrument", "no_vocal", "no vocal")
        )

        candidates = [
            Path(name) for name in output_files
            if any(word in Path(name).name.lower()
                   for word in keywords)
        ]

        if not candidates:
            raise RuntimeError(
                f"Could not identify the {options} output. "
                "The selected model may use different stem names."
            )

        source = candidates[0]
        destination = OUTPUT_DIR / (
            f"{source.stem}_{os.urandom(4).hex()}.wav"
        )
        shutil.copy2(source, destination)

        return FileResponse(
            path=destination,
            media_type="audio/wav",
            filename=f"uvr_{options}.wav"
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            500,
            f"Audio separation failed: {str(exc)}"
        ) from exc
    finally:
        await file.close()
        shutil.rmtree(workdir, ignore_errors=True)
      
