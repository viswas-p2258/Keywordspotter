import io

import pytest

from app import app


@pytest.fixture
def client():
    app.config.update(
        TESTING=True,
        SECRET_KEY="test-secret",
        UPLOAD_BUCKET=None,
        TRANSCRIBE_DATA_ACCESS_ROLE_ARN=None,
        MAX_UPLOAD_BYTES=64 * 1024 * 1024,
        MAX_CONTENT_LENGTH=64 * 1024 * 1024,
    )
    with app.test_client() as test_client:
        yield test_client


def test_health_endpoint_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_homepage_renders(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b"Speech Transcriber" in response.data


def test_homepage_announces_amazon_transcribe_when_configured(client):
    app.config.update(
        UPLOAD_BUCKET="test-audio",
        TRANSCRIBE_DATA_ACCESS_ROLE_ARN="arn:aws:iam::123456789012:role/Transcribe",
    )

    response = client.get("/")

    assert response.status_code == 200
    assert b"transcribed by Amazon Transcribe" in response.data


def test_transcribe_without_file_shows_validation_message(client):
    response = client.post("/transcribe", data={})

    assert response.status_code == 200
    assert b"choose a WAV audio file" in response.data


def test_transcribe_displays_recognized_speech(client, monkeypatch):
    monkeypatch.setattr("app.transcribe_audio", lambda audio_path: "Turn left at the next street.")
    response = client.post(
        "/transcribe",
        data={"file": (io.BytesIO(b"RIFF fake wav data"), "speech.wav")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"Turn left at the next street." in response.data
    assert b"id=\"transcription\"" in response.data


def test_transcribe_reports_empty_speech(client, monkeypatch):
    monkeypatch.setattr("app.transcribe_audio", lambda audio_path: "")
    response = client.post(
        "/transcribe",
        data={"file": (io.BytesIO(b"RIFF fake wav data"), "silence.wav")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"No speech was detected in this audio." in response.data


def test_transcribe_rejects_non_wav_file(client):
    response = client.post(
        "/transcribe",
        data={"file": (io.BytesIO(b"not a wav"), "speech.mp3")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"Please upload a .wav file" in response.data


def test_s3_presign_rejects_oversized_upload(client):
    app.config.update(UPLOAD_BUCKET="test-audio", MAX_UPLOAD_BYTES=10)

    response = client.post(
        "/uploads/presign",
        json={"filename": "speech.wav", "size": 11},
    )

    assert response.status_code == 413
    assert response.get_json() == {"error": "Audio file exceeds the maximum upload size."}


def test_s3_presign_returns_temporary_upload_url(client, monkeypatch):
    app.config.update(UPLOAD_BUCKET="test-audio", MAX_UPLOAD_BYTES=64 * 1024 * 1024)

    class FakeS3:
        def generate_presigned_post(self, bucket, key, *, Fields, Conditions, ExpiresIn):
            assert bucket == "test-audio"
            assert key.startswith("uploads/")
            assert Fields["Content-Type"] == "audio/wav"
            assert {"Content-Type": "audio/wav"} in Conditions
            assert ["content-length-range", 1, 64 * 1024 * 1024] in Conditions
            assert ExpiresIn == 300
            return {"url": "https://s3.example/upload", "fields": {"key": key}}

    monkeypatch.setattr("app.boto3.client", lambda service: FakeS3())
    response = client.post(
        "/uploads/presign",
        json={"filename": "speech.wav", "size": 100},
    )

    assert response.status_code == 200
    assert response.get_json()["upload_url"] == "https://s3.example/upload"
    assert response.get_json()["upload_fields"]["key"].startswith("uploads/")
    assert response.get_json()["object_key"].startswith("uploads/")


def test_s3_transcription_deletes_temporary_audio(client, monkeypatch):
    app.config.update(UPLOAD_BUCKET="test-audio")
    deleted = []

    class FakeS3:
        def download_file(self, bucket, key, path):
            assert bucket == "test-audio"
            assert key == "uploads/abc.wav"

        def head_object(self, *, Bucket, Key):
            assert Bucket == "test-audio"
            assert Key == "uploads/abc.wav"
            return {"ContentLength": 100}

        def delete_object(self, *, Bucket, Key):
            deleted.append((Bucket, Key))

    monkeypatch.setattr("app.boto3.client", lambda service: FakeS3())
    monkeypatch.setattr("app.transcribe_audio", lambda audio_path: "recognized speech")
    response = client.post("/transcribe", json={"object_key": "uploads/abc.wav"})

    assert response.status_code == 200
    assert response.get_json() == {"transcription": "recognized speech"}
    assert deleted == [("test-audio", "uploads/abc.wav")]


def test_s3_transcription_rejects_oversized_object(client, monkeypatch):
    app.config.update(UPLOAD_BUCKET="test-audio", MAX_UPLOAD_BYTES=10)
    deleted = []

    class FakeS3:
        def head_object(self, *, Bucket, Key):
            return {"ContentLength": 11}

        def download_file(self, bucket, key, path):
            pytest.fail("Oversized upload should not be downloaded.")

        def delete_object(self, *, Bucket, Key):
            deleted.append((Bucket, Key))

    monkeypatch.setattr("app.boto3.client", lambda service: FakeS3())
    response = client.post("/transcribe", json={"object_key": "uploads/abc.wav"})

    assert response.status_code == 413
    assert response.get_json() == {"error": "Audio file exceeds the maximum upload size."}
    assert deleted == [("test-audio", "uploads/abc.wav")]


def test_s3_transcribe_starts_amazon_transcribe_job(client, monkeypatch):
    app.config.update(
        UPLOAD_BUCKET="test-audio",
        TRANSCRIBE_DATA_ACCESS_ROLE_ARN="arn:aws:iam::123456789012:role/Transcribe",
        TRANSCRIBE_LANGUAGE_CODE="en-IN",
    )
    started = {}

    class FakeTranscribe:
        def start_transcription_job(self, **kwargs):
            started.update(kwargs)

    monkeypatch.setattr("app.boto3.client", lambda service: FakeTranscribe())
    response = client.post(
        "/transcribe",
        json={"object_key": "uploads/0123456789abcdef0123456789abcdef.wav"},
    )

    assert response.status_code == 202
    job_id = response.get_json()["job_id"]
    assert len(job_id) == 32
    assert started["TranscriptionJobName"] == job_id
    assert started["LanguageCode"] == "en-IN"
    assert started["MediaFormat"] == "wav"
    assert started["Media"]["MediaFileUri"] == (
        "s3://test-audio/uploads/0123456789abcdef0123456789abcdef.wav"
    )
    assert started["OutputBucketName"] == "test-audio"
    assert started["OutputKey"] == f"transcripts/{job_id}.json"
    assert started["JobExecutionSettings"]["AllowDeferredExecution"] is True
    assert started["JobExecutionSettings"]["DataAccessRoleArn"].endswith(":role/Transcribe")


def test_amazon_transcribe_status_returns_result_and_removes_audio(client, monkeypatch):
    app.config["UPLOAD_BUCKET"] = "test-audio"
    deleted = []

    class FakeTranscribe:
        def get_transcription_job(self, *, TranscriptionJobName):
            return {"TranscriptionJob": {"TranscriptionJobStatus": "COMPLETED"}}

    class FakeBody:
        def read(self):
            return b'{"results":{"transcripts":[{"transcript":"hello world"}]}}'

    class FakeS3:
        def get_object(self, *, Bucket, Key):
            assert Bucket == "test-audio"
            assert Key == "transcripts/0123456789abcdef0123456789abcdef.json"
            return {"Body": FakeBody()}

        def delete_object(self, *, Bucket, Key):
            deleted.append((Bucket, Key))

    monkeypatch.setattr(
        "app.boto3.client",
        lambda service: FakeTranscribe() if service == "transcribe" else FakeS3(),
    )
    response = client.get("/transcribe/status/0123456789abcdef0123456789abcdef")

    assert response.status_code == 200
    assert response.get_json() == {
        "status": "COMPLETED",
        "transcription": "hello world",
    }
    assert deleted == [("test-audio", "uploads/0123456789abcdef0123456789abcdef.wav")]


def test_amazon_transcribe_status_returns_in_progress(client, monkeypatch):
    class FakeTranscribe:
        def get_transcription_job(self, *, TranscriptionJobName):
            return {"TranscriptionJob": {"TranscriptionJobStatus": "IN_PROGRESS"}}

    monkeypatch.setattr("app.boto3.client", lambda service: FakeTranscribe())
    response = client.get("/transcribe/status/0123456789abcdef0123456789abcdef")

    assert response.status_code == 202
    assert response.get_json() == {"status": "IN_PROGRESS"}


def test_amazon_transcribe_status_cleans_audio_after_failed_job(client, monkeypatch):
    app.config["UPLOAD_BUCKET"] = "test-audio"
    deleted = []

    class FakeTranscribe:
        def get_transcription_job(self, *, TranscriptionJobName):
            return {
                "TranscriptionJob": {
                    "TranscriptionJobStatus": "FAILED",
                    "FailureReason": "Unsupported media format.",
                }
            }

    class FakeS3:
        def delete_object(self, *, Bucket, Key):
            deleted.append((Bucket, Key))

    monkeypatch.setattr(
        "app.boto3.client",
        lambda service: FakeTranscribe() if service == "transcribe" else FakeS3(),
    )
    response = client.get("/transcribe/status/0123456789abcdef0123456789abcdef")

    assert response.status_code == 422
    assert response.get_json()["status"] == "FAILED"
    assert deleted == [("test-audio", "uploads/0123456789abcdef0123456789abcdef.wav")]


def test_amazon_transcribe_status_rejects_invalid_job_id(client):
    response = client.get("/transcribe/status/invalid")

    assert response.status_code == 400
    assert response.get_json() == {"error": "The transcription job reference is invalid."}


def test_s3_transcribe_handles_botocore_error(client, monkeypatch):
    from botocore.exceptions import ParamValidationError

    app.config.update(
        UPLOAD_BUCKET="test-audio",
        TRANSCRIBE_DATA_ACCESS_ROLE_ARN="arn:aws:iam::123456789012:role/Transcribe",
    )

    class FailingTranscribe:
        def start_transcription_job(self, **kwargs):
            raise ParamValidationError(report="Invalid parameter")

    class FakeS3:
        def delete_object(self, *, Bucket, Key):
            pass

    monkeypatch.setattr(
        "app.boto3.client",
        lambda service: FailingTranscribe() if service == "transcribe" else FakeS3(),
    )
    response = client.post(
        "/transcribe",
        json={"object_key": "uploads/0123456789abcdef0123456789abcdef.wav"},
    )

    assert response.status_code == 503
    assert response.get_json() == {"error": "Amazon Transcribe could not start this audio job."}
