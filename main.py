import os
import json
from datetime import datetime

import numpy as np
import requests
import librosa
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from PIL import Image
from pydub import AudioSegment


app = Flask(__name__)

#==============================================================
# API KEY DE XENO-CANTO Y CONFIGURACION DE DESCARGA
#==============================================================
XENOCANTO_API_KEY = "bfc619e9677e987f549863f8a7184c6c17f8f636"
XENOCANTO_API_V3  = "https://xeno-canto.org/api/3/recordings"
#==============================================================

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

UPLOAD_DIR   = os.path.join("static", "uploads")
HISTORY_FILE = "historial.json"
MODEL_PATH   = "modelo_aves_real.h5"

os.makedirs(UPLOAD_DIR, exist_ok=True)

SPECIES_INDEX = {
    0: {"common_name": "Bichofue Gritón",     "scientific_name": "Pitangus sulphuratus"},
    1: {"common_name": "Siriri Comun",        "scientific_name": "Tyrannus melancholicus"},
    2: {"common_name": "Azulejo de Palma",    "scientific_name": "Thraupis palmarum"},
    3: {"common_name": "Azulejo Comun",       "scientific_name": "Thraupis episcopus"},
    4: {"common_name": "Copeton Andino",      "scientific_name": "Zonotrichia capensis"},
    5: {"common_name": "Mirla Embarradora",   "scientific_name": "Turdus ignobilis"},
    6: {"common_name": "Cucarachero Comun",   "scientific_name": "Troglodytes aedon"},
    7: {"common_name": "Torcaza Naguiblanca", "scientific_name": "Zenaida auriculata"},
    8: {"common_name": "Colibri Amazilia",    "scientific_name": "Amazilia tzacatl"},
    9: {"common_name": "Garza del Ganado",    "scientific_name": "Bubulcus ibis"},
}

if os.path.exists(MODEL_PATH):
    inference_model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Modelo CNN cargado desde '{MODEL_PATH}'.")
else:
    inference_model = None
    print(f"Archivo '{MODEL_PATH}' no encontrado.")


def load_analysis_history() -> list[dict]:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as exc:
        print(f"No se pudo leer el historial: {exc}")
        return []


def append_to_history(new_record: dict) -> None:
    history = load_analysis_history()
    history.insert(0, new_record)
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4, ensure_ascii=False)
    except IOError as exc:
        print(f"No se pudo escribir el historial: {exc}")


def fetch_xenocanto_references(scientific_name: str) -> list[dict]:
    try:
        params = {
            "query": f'sp:"{scientific_name}"',
            "key":   XENOCANTO_API_KEY,
        }
        response = requests.get(
            XENOCANTO_API_V3,
            params=params,
            headers=REQUEST_HEADERS,
            timeout=4,
        )
        response.raise_for_status()

        recordings = response.json().get("recordings", [])

        return [
            {
                "id":        rec.get("id"),
                "country":   rec.get("cnt"),
                "locality":  rec.get("loc"),
                "recorder":  rec.get("rec"),
                "audio_url": rec.get("file"),
            }
            for rec in recordings[:1]
        ]

    except requests.RequestException as exc:
        print(f"Error al consultar Xeno-canto: {exc}")
        return []


def process_and_classify_audio(
    audio_path: str,
) -> tuple[str, str, str, list[dict]]:
    clean_wav_path = audio_path + "_normalized.wav"

    predicted_class_id = None
    confidence_score   = 0.0

    try:
        try:
            waveform, sr = librosa.load(audio_path, sr=16000, mono=True, duration=5)
        except Exception as load_exc:
            print(f"librosa.load fallo: {load_exc}. Usando fallback PyDub...")
            audio_segment = (
                AudioSegment.from_file(audio_path)
                .set_frame_rate(16000)
                .set_channels(1)
                .set_sample_width(2)[:5000]
            )
            audio_segment.export(clean_wav_path, format="wav")
            import scipy.io.wavfile as wav_reader
            sr, samples = wav_reader.read(clean_wav_path)
            waveform = samples.astype(np.float32) / 32768.0

        mel_spectrogram = librosa.feature.melspectrogram(
            y=waveform, sr=sr, n_mels=128, n_fft=2048,
            hop_length=512, fmin=50, fmax=8000,
        )
        mel_db      = librosa.power_to_db(mel_spectrogram, ref=np.max, top_db=80)
        mel_db_norm = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

        mel_db_norm_t = mel_db_norm.T
        mel_min       = mel_db_norm_t.min()
        mel_max       = mel_db_norm_t.max() + 1e-6
        mel_scaled    = ((mel_db_norm_t - mel_min) / (mel_max - mel_min) * 255).astype(np.uint8)

        img_pil      = Image.fromarray(mel_scaled, mode="L").resize((224, 224), Image.Resampling.LANCZOS)
        input_tensor = np.stack([np.array(img_pil)] * 3, axis=-1) / 255.0
        input_batch  = np.expand_dims(input_tensor, axis=0)

        if inference_model is not None:
            prediction         = inference_model.predict(input_batch, verbose=0)[0]
            predicted_class_id = int(np.argmax(prediction))
            confidence_score   = float(prediction[predicted_class_id]) * 100
        else:
            predicted_class_id = 0
            confidence_score   = 0.0

    except Exception as exc:
        print(f"Fallo en el procesamiento de audio: {exc}")
        predicted_class_id = 0
        confidence_score   = 0.0

    finally:
        if os.path.exists(clean_wav_path):
            os.remove(clean_wav_path)

    species_info         = SPECIES_INDEX.get(predicted_class_id, {"common_name": "Especie no identificada", "scientific_name": "Incierto"})
    reference_recordings = fetch_xenocanto_references(species_info["scientific_name"])
    formatted_confidence = f"{round(confidence_score, 2)}%"

    return (
        species_info["common_name"],
        species_info["scientific_name"],
        formatted_confidence,
        reference_recordings,
    )


@app.route("/")
def home():
    history = load_analysis_history()
    return render_template("index.html", historial=history)


@app.route("/subir_audio", methods=["POST"])
def upload_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No se recibio ningun archivo de audio."}), 400

    uploaded_file = request.files["audio"]

    if not uploaded_file.filename:
        return jsonify({"error": "El archivo no contiene nombre valido."}), 400

    timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_filename = f"audio_{timestamp}.wav"
    audio_filepath = os.path.join(UPLOAD_DIR, audio_filename)

    uploaded_file.save(audio_filepath)

    try:
        common_name, scientific_name, confidence, xc_references = (
            process_and_classify_audio(audio_filepath)
        )
    except Exception as exc:
        return jsonify({"error": f"Fallo durante el procesamiento: {str(exc)}"}), 500

    analysis_record = {
        "fecha":             datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "archivo":           f"/static/uploads/{audio_filename}",
        "resultado":         common_name,
        "nombre_cientifico": scientific_name,
        "confianza":         confidence,
        "xeno_canto":        xc_references,
    }

    append_to_history(analysis_record)
    return jsonify(analysis_record)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
