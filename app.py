#!/usr/bin/env python3

"""
@author: Jithin Sasikumar

Script to create a web application that wraps the trained model to be used for inference using
`FLASK API`. It facilitates the application to run from a server which defines every routes and
functions to perform. The front-end is designed using `./templates/page.html` and its styles in
`./static/page.css`

Note:
    Make sure to define all the variables and valid paths in `.config_dir/config.yaml` to run
    this script without errors and issues.
"""

import os
import tempfile
from flask import Flask, render_template, request, flash
from src.transcription import TranscriptionError, transcribe_audio

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY") or os.urandom(32)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

@app.route('/')
def home():
    """
    Returns the result of calling render_template() with page.html
    """
    return render_template('page.html')

@app.route('/health', methods=['GET'])
def health():
    """Simple health check endpoint for load balancers and deployment checks."""
    return {"status": "ok"}, 200


@app.errorhandler(413)
def request_too_large(error):
    app.logger.warning("Rejected oversized audio upload: %s", error)
    flash("Audio file is too large. Please choose a WAV file smaller than 64 MB.", category="error")
    return render_template("page.html"), 413


@app.route("/transcribe", methods = ["POST"])
def transcribe():
    """Transcribe the uploaded WAV file and show its recognized words."""
    audio_file = request.files.get("file")
    if audio_file is None or not audio_file.filename:
        flash("Please choose a WAV audio file.", category="error")
        return render_template("page.html")

    if not audio_file.filename.lower().endswith(".wav"):
        flash("Unsupported file format. Please upload a .wav file.", category="error")
        return render_template("page.html")

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary_file:
            temporary_path = temporary_file.name
            audio_file.save(temporary_file)

        transcription = transcribe_audio(temporary_path)
        if not transcription:
            transcription = "No speech was detected in this audio."
        return render_template("page.html", transcription=transcription)
    except TranscriptionError as exc:
        app.logger.warning("Audio transcription failed: %s", exc)
        flash(str(exc), category="error")
        return render_template("page.html")
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
    )