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

import json
import os
import re
import tempfile
import uuid

import boto3
from botocore.exceptions import ClientError
from flask import Flask, render_template, request, flash
from src.transcription import TranscriptionError, transcribe_audio

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY") or os.urandom(32)
app.config["MAX_UPLOAD_BYTES"] = int(os.environ.get("MAX_UPLOAD_BYTES", str(64 * 1024 * 1024)))
app.config["MAX_CONTENT_LENGTH"] = app.config["MAX_UPLOAD_BYTES"]
app.config["UPLOAD_BUCKET"] = os.environ.get("UPLOAD_BUCKET")
app.config["TRANSCRIBE_DATA_ACCESS_ROLE_ARN"] = os.environ.get("TRANSCRIBE_DATA_ACCESS_ROLE_ARN")
app.config["TRANSCRIBE_LANGUAGE_CODE"] = os.environ.get("TRANSCRIBE_LANGUAGE_CODE", "en-US")


def _upload_bucket() -> str:
    bucket = app.config.get("UPLOAD_BUCKET")
    if not bucket:
        raise RuntimeError("UPLOAD_BUCKET must be configured for S3 uploads.")
    return bucket

@app.route('/')
def home():
    """
    Returns the result of calling render_template() with page.html
    """
    return render_template(
        'page.html',
        s3_uploads_enabled=bool(app.config.get("UPLOAD_BUCKET")),
        amazon_transcribe_enabled=bool(
            app.config.get("UPLOAD_BUCKET") and app.config.get("TRANSCRIBE_DATA_ACCESS_ROLE_ARN")
        ),
        max_upload_bytes=app.config["MAX_UPLOAD_BYTES"],
    )

@app.route('/health', methods=['GET'])
def health():
    """Simple health check endpoint for load balancers and deployment checks."""
    return {"status": "ok"}, 200


@app.route("/uploads/presign", methods=["POST"])
def presign_upload():
    """Create a short-lived URL for uploading an audio file directly to private S3."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return {"error": "A JSON upload request is required."}, 400

    filename = payload.get("filename")
    size = payload.get("size")
    if not isinstance(filename, str) or not filename.lower().endswith(".wav"):
        return {"error": "Please choose a WAV audio file."}, 400
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        return {"error": "The audio file size is invalid."}, 400
    if size > app.config["MAX_UPLOAD_BYTES"]:
        return {"error": "Audio file exceeds the maximum upload size."}, 413

    object_key = f"uploads/{uuid.uuid4().hex}.wav"
    try:
        bucket = _upload_bucket()
        upload = boto3.client("s3").generate_presigned_post(
            bucket,
            object_key,
            Fields={"Content-Type": "audio/wav"},
            Conditions=[
                {"Content-Type": "audio/wav"},
                ["content-length-range", 1, app.config["MAX_UPLOAD_BYTES"]],
            ],
            ExpiresIn=300,
        )
    except (ClientError, RuntimeError) as exc:
        app.logger.error("Could not create a presigned audio upload URL: %s", exc)
        return {"error": "Audio uploads are temporarily unavailable."}, 503

    return {
        "upload_url": upload["url"],
        "upload_fields": upload["fields"],
        "object_key": object_key,
    }


def _delete_s3_upload(object_key: str) -> None:
    try:
        boto3.client("s3").delete_object(Bucket=_upload_bucket(), Key=object_key)
    except (ClientError, RuntimeError) as exc:
        app.logger.warning("Could not remove temporary S3 audio upload: %s", exc)


@app.errorhandler(413)
def request_too_large(error):
    app.logger.warning("Rejected oversized audio upload: %s", error)
    flash("Audio file is too large. Please choose a WAV file smaller than 64 MB.", category="error")
    return render_template(
        "page.html",
        s3_uploads_enabled=bool(app.config.get("UPLOAD_BUCKET")),
        amazon_transcribe_enabled=bool(
            app.config.get("UPLOAD_BUCKET") and app.config.get("TRANSCRIBE_DATA_ACCESS_ROLE_ARN")
        ),
        max_upload_bytes=app.config["MAX_UPLOAD_BYTES"],
    ), 413


def _start_amazon_transcribe_job(object_key: str):
    """Start an asynchronous Amazon Transcribe job for a private S3 audio upload."""
    match = re.fullmatch(r"uploads/([a-f0-9]{32})\.wav", object_key)
    if not match:
        return {"error": "The uploaded audio reference is invalid."}, 400

    role_arn = app.config.get("TRANSCRIBE_DATA_ACCESS_ROLE_ARN")
    if not role_arn:
        return {"error": "Amazon Transcribe is not configured."}, 503

    job_name = match.group(1)
    try:
        boto3.client("transcribe").start_transcription_job(
            TranscriptionJobName=job_name,
            LanguageCode=app.config["TRANSCRIBE_LANGUAGE_CODE"],
            MediaFormat="wav",
            Media={"MediaFileUri": f"s3://{_upload_bucket()}/{object_key}"},
            OutputBucketName=_upload_bucket(),
            OutputKey=f"transcripts/{job_name}.json",
            DataAccessRoleArn=role_arn,
        )
    except (ClientError, RuntimeError) as exc:
        app.logger.error("Could not start Amazon Transcribe job: %s", exc)
        _delete_s3_upload(object_key)
        return {"error": "Amazon Transcribe could not start this audio job."}, 503

    return {"job_id": job_name, "status": "QUEUED"}, 202


@app.route("/transcribe/status/<job_id>", methods=["GET"])
def transcribe_status(job_id: str):
    """Return the status of an Amazon Transcribe job and its result when complete."""
    if not re.fullmatch(r"[a-f0-9]{32}", job_id):
        return {"error": "The transcription job reference is invalid."}, 400

    try:
        job = boto3.client("transcribe").get_transcription_job(
            TranscriptionJobName=job_id
        )["TranscriptionJob"]
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code == "NotFoundException":
            return {"error": "The transcription job was not found."}, 404
        app.logger.error("Could not retrieve Amazon Transcribe job status: %s", exc)
        return {"error": "Amazon Transcribe status is temporarily unavailable."}, 503

    status = job["TranscriptionJobStatus"]
    if status in {"QUEUED", "IN_PROGRESS"}:
        return {"status": status}, 202

    object_key = f"uploads/{job_id}.wav"
    if status == "FAILED":
        _delete_s3_upload(object_key)
        app.logger.warning(
            "Amazon Transcribe job %s failed: %s",
            job_id,
            job.get("FailureReason", "No failure reason was provided."),
        )
        return {
            "status": "FAILED",
            "error": "Amazon Transcribe could not process this WAV file.",
        }, 422

    if status != "COMPLETED":
        app.logger.error("Amazon Transcribe returned unexpected job status: %s", status)
        return {"error": "Amazon Transcribe returned an unknown job status."}, 502

    try:
        transcript_object = boto3.client("s3").get_object(
            Bucket=_upload_bucket(),
            Key=f"transcripts/{job_id}.json",
        )
        transcript_data = json.loads(transcript_object["Body"].read())
        transcript = transcript_data["results"]["transcripts"][0]["transcript"]
    except (ClientError, KeyError, IndexError, TypeError, ValueError) as exc:
        app.logger.error("Could not read Amazon Transcribe result for job %s: %s", job_id, exc)
        return {"error": "The transcription result is temporarily unavailable."}, 503
    finally:
        _delete_s3_upload(object_key)

    return {"status": "COMPLETED", "transcription": transcript or "No speech was detected."}


def _transcribe_s3_upload(object_key: str):
    """Download, transcribe, and remove an uploaded S3 object."""
    if not object_key.startswith("uploads/") or ".." in object_key:
        return {"error": "The uploaded audio reference is invalid."}, 400

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary_file:
            temporary_path = temporary_file.name

        s3 = boto3.client("s3")
        object_size = s3.head_object(Bucket=_upload_bucket(), Key=object_key)["ContentLength"]
        if object_size > app.config["MAX_UPLOAD_BYTES"]:
            return {"error": "Audio file exceeds the maximum upload size."}, 413
        s3.download_file(_upload_bucket(), object_key, temporary_path)
        transcription = transcribe_audio(temporary_path)
        return {"transcription": transcription or "No speech was detected in this audio."}, 200
    except ClientError as exc:
        app.logger.error("Could not retrieve uploaded audio from S3: %s", exc)
        return {"error": "Could not retrieve the uploaded audio. Please upload it again."}, 400
    except (RuntimeError, TranscriptionError) as exc:
        app.logger.warning("Audio transcription failed: %s", exc)
        return {"error": str(exc)}, 422
    finally:
        try:
            boto3.client("s3").delete_object(Bucket=_upload_bucket(), Key=object_key)
        except (ClientError, RuntimeError) as exc:
            app.logger.warning("Could not remove temporary S3 audio upload: %s", exc)
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)


@app.route("/transcribe", methods = ["POST"])
def transcribe():
    """Transcribe the uploaded WAV file and show its recognized words."""
    if request.is_json:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("object_key"), str):
            return {"error": "An uploaded audio reference is required."}, 400
        if app.config.get("TRANSCRIBE_DATA_ACCESS_ROLE_ARN"):
            return _start_amazon_transcribe_job(payload["object_key"])
        return _transcribe_s3_upload(payload["object_key"])

    audio_file = request.files.get("file")
    if audio_file is None or not audio_file.filename:
        flash("Please choose a WAV audio file.", category="error")
        return render_template(
            "page.html",
            s3_uploads_enabled=bool(app.config.get("UPLOAD_BUCKET")),
            amazon_transcribe_enabled=bool(
                app.config.get("UPLOAD_BUCKET") and app.config.get("TRANSCRIBE_DATA_ACCESS_ROLE_ARN")
            ),
            max_upload_bytes=app.config["MAX_UPLOAD_BYTES"],
        )

    if not audio_file.filename.lower().endswith(".wav"):
        flash("Unsupported file format. Please upload a .wav file.", category="error")
        return render_template(
            "page.html",
            s3_uploads_enabled=bool(app.config.get("UPLOAD_BUCKET")),
            amazon_transcribe_enabled=bool(
                app.config.get("UPLOAD_BUCKET") and app.config.get("TRANSCRIBE_DATA_ACCESS_ROLE_ARN")
            ),
            max_upload_bytes=app.config["MAX_UPLOAD_BYTES"],
        )

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary_file:
            temporary_path = temporary_file.name
            audio_file.save(temporary_file)

        transcription = transcribe_audio(temporary_path)
        if not transcription:
            transcription = "No speech was detected in this audio."
        return render_template(
            "page.html",
            transcription=transcription,
            s3_uploads_enabled=bool(app.config.get("UPLOAD_BUCKET")),
            amazon_transcribe_enabled=bool(
                app.config.get("UPLOAD_BUCKET") and app.config.get("TRANSCRIBE_DATA_ACCESS_ROLE_ARN")
            ),
            max_upload_bytes=app.config["MAX_UPLOAD_BYTES"],
        )
    except TranscriptionError as exc:
        app.logger.warning("Audio transcription failed: %s", exc)
        flash(str(exc), category="error")
        return render_template(
            "page.html",
            s3_uploads_enabled=bool(app.config.get("UPLOAD_BUCKET")),
            amazon_transcribe_enabled=bool(
                app.config.get("UPLOAD_BUCKET") and app.config.get("TRANSCRIBE_DATA_ACCESS_ROLE_ARN")
            ),
            max_upload_bytes=app.config["MAX_UPLOAD_BYTES"],
        )
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
    )