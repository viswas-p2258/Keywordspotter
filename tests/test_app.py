import io

import pytest

from app import app


@pytest.fixture
def client():
    app.config.update(TESTING=True, SECRET_KEY="test-secret")
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
