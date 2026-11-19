import os, csv, json, tempfile
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = Path(os.environ.get("DYSARTHRIA_DATASET_DIR", BASE_DIR / "dataset"))
ALLOWED = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4"}
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")
_model = None
_model_error = None

def get_model():
    global _model, _model_error
    if _model is not None:
        return _model
    if _model_error:
        raise RuntimeError(_model_error)
    try:
        from faster_whisper import WhisperModel
        # CPU mode works on ordinary laptops; set WHISPER_DEVICE=cuda if configured.
        device = os.environ.get("WHISPER_DEVICE", "cpu")
        compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "int8" if device == "cpu" else "float16")
        _model = WhisperModel(MODEL_SIZE, device=device, compute_type=compute_type)
        return _model
    except Exception as exc:
        _model_error = ("Could not load the speech-recognition model. "
                        "Check installation/internet for the first model download. Details: " + str(exc))
        raise RuntimeError(_model_error)

def list_dataset_samples():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for p in sorted(DATASET_DIR.rglob("*")):
        if p.is_file() and p.suffix.lower() in ALLOWED:
            items.append({"id": str(p.relative_to(DATASET_DIR)), "name": p.name})
    return items

@app.route("/")
def home():
    return render_template("index.html", model=MODEL_SIZE)

@app.route("/api/status")
def status():
    return jsonify({"status": "ready", "model": MODEL_SIZE, "dataset_samples": len(list_dataset_samples())})

@app.route("/api/samples")
def samples():
    return jsonify(list_dataset_samples())

@app.route("/api/sample/<path:sample_id>")
def sample(sample_id):
    # Only serve files found inside the configured dataset directory.
    target = (DATASET_DIR / sample_id).resolve()
    if DATASET_DIR.resolve() not in target.parents or not target.is_file():
        return jsonify({"error": "Sample not found"}), 404
    return send_from_directory(target.parent, target.name)

@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    audio = request.files.get("audio")
    if not audio or not audio.filename:
        return jsonify({"error": "Please record or select an audio file first."}), 400
    suffix = Path(secure_filename(audio.filename)).suffix.lower() or ".wav"
    if suffix not in ALLOWED:
        return jsonify({"error": "Unsupported audio format. Use WAV, MP3, M4A, FLAC, OGG or WEBM."}), 400
    language = request.form.get("language", "en").strip() or None
    profile_json = request.form.get("profile", "{}")
    try:
        corrections = json.loads(profile_json)
        if not isinstance(corrections, dict):
            corrections = {}
    except Exception:
        corrections = {}
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            audio.save(tmp.name)
            tmp_path = tmp.name
        model = get_model()
        segments, info = model.transcribe(
            tmp_path, language=language, beam_size=5, vad_filter=True,
            word_timestamps=False
        )
        segments = list(segments)
        raw_text = " ".join(s.text.strip() for s in segments).strip()
        if not raw_text:
            return jsonify({"error": "No clear speech was detected. Try a clearer or longer recording."}), 422
        # Personalization uses the user's own saved phrase/word corrections.
        personalized_text = raw_text
        applied = []
        for wrong, right in corrections.items():
            wrong, right = str(wrong).strip(), str(right).strip()
            if wrong and right and wrong.lower() in personalized_text.lower():
                import re
                personalized_text, count = re.subn(re.escape(wrong), right, personalized_text, flags=re.I)
                if count:
                    applied.append({"from": wrong, "to": right})
        # A transparent heuristic, not a clinically validated probability.
        if segments:
            avg_logprob = sum(float(s.avg_logprob) for s in segments) / len(segments)
            no_speech = sum(float(s.no_speech_prob) for s in segments) / len(segments)
            confidence = max(0, min(99, round((1 / (1 + __import__("math").exp(-((avg_logprob + 1.0) * 2.2)))) * 100 * (1 - no_speech))))
        else:
            confidence = 0
        needs_clarification = confidence < 65 or len(personalized_text.split()) <= 1
        return jsonify({
            "transcript": raw_text,
            "personalized_text": personalized_text,
            "confidence": confidence,
            "needs_clarification": needs_clarification,
            "clarification_message": ("Please check the recognized message before using it." if needs_clarification else ""),
            "language": getattr(info, "language", language or "unknown"),
            "applied_personalization": applied,
            "segments": len(segments),
            "confidence_note": "Estimated from speech-model signals for prototype feedback; not a calibrated clinical probability."
        })
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": "Audio processing failed: " + str(exc)}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=True)
