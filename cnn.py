import os
import time
import requests
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image
from pydub import AudioSegment

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#==============================================================
# API KEY DE XENO-CANTO Y CONFIGURACION DE DESCARGA
#==============================================================
XENOCANTO_API_KEY = "     " # Pegar Api Key aqui
XENOCANTO_API_V3  = "https://xeno-canto.org/api/3/recordings"
#==============================================================

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

DELAY_BETWEEN_SPECIES = 4
DELAY_BETWEEN_FILES   = 0.5
MAX_RETRIES           = 3
RETRY_BACKOFF         = 10

SPECIES_CATALOG = [
    {"common_name": "Bichofue Gritón",        "scientific_name": "Pitangus sulphuratus"},
    {"common_name": "Siriri Comun",           "scientific_name": "Tyrannus melancholicus"},
    {"common_name": "Azulejo de Palma",       "scientific_name": "Thraupis palmarum"},
    {"common_name": "Azulejo Comun",          "scientific_name": "Thraupis episcopus"},
    {"common_name": "Copeton Andino",         "scientific_name": "Zonotrichia capensis"},
    {"common_name": "Mirla Embarradora",      "scientific_name": "Turdus ignobilis"},
    {"common_name": "Cucarachero Comun",      "scientific_name": "Troglodytes aedon"},
    {"common_name": "Torcaza Naguiblanca",    "scientific_name": "Zenaida auriculata"},
    {"common_name": "Colibri Amazilia",       "scientific_name": "Amazilia tzacatl"},
    {"common_name": "Garza del Ganado",       "scientific_name": "Bubulcus ibis"},
]

NUM_CLASSES         = len(SPECIES_CATALOG)
SAMPLES_PER_SPECIES = 40
IMAGE_SIZE          = (224, 224)
TEMP_DATASET_DIR    = "dataset_temp"
AUGMENTATION_THRESHOLD = 35

os.makedirs(TEMP_DATASET_DIR, exist_ok=True)


def download_species_recordings(scientific_name: str, max_recordings: int) -> list[str]:
    species_dir = os.path.join(TEMP_DATASET_DIR, scientific_name.replace(" ", "_"))
    os.makedirs(species_dir, exist_ok=True)

    cached_files = [
        os.path.join(species_dir, f)
        for f in os.listdir(species_dir)
        if f.endswith(".mp3")
    ]
    if len(cached_files) >= max_recordings:
        print(f"   Datos ya descargados para '{scientific_name}'. Se omite descarga remota.")
        return cached_files

    print(f"   Consultando Xeno-canto v3 para: {scientific_name} ...")

    params = {
        "query": f'sp:"{scientific_name}"',
        "key":   XENOCANTO_API_KEY,
    }

    recordings = []
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            api_response = requests.get(
                XENOCANTO_API_V3, params=params, headers=REQUEST_HEADERS, timeout=20,
            )
            api_response.raise_for_status()
            recordings = api_response.json().get("recordings", [])[:max_recordings]
            print(f"   {len(recordings)} grabaciones encontradas.")
            break
        except requests.RequestException as exc:
            wait = RETRY_BACKOFF * attempt
            print(f"   [REINTENTO {attempt}/{MAX_RETRIES}] {exc}. Esperando {wait}s...")
            time.sleep(wait)

    if not recordings:
        print(f"   Sin grabaciones para '{scientific_name}' tras {MAX_RETRIES} intentos.")
        return cached_files

    for index, recording in enumerate(recordings):
        audio_url   = recording.get("file")
        destination = os.path.join(species_dir, f"recording_{index:03d}.mp3")

        if os.path.exists(destination) or not audio_url:
            continue

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                audio_response = requests.get(
                    audio_url, headers=REQUEST_HEADERS, timeout=25, allow_redirects=True,
                )
                if audio_response.status_code == 200:
                    with open(destination, "wb") as audio_file:
                        audio_file.write(audio_response.content)
                    print(f"      Archivo {index + 1}/{len(recordings)} descargado.")
                    break
                else:
                    print(f"      HTTP {audio_response.status_code} para archivo {index + 1}. Se omite.")
                    break
            except requests.RequestException as exc:
                wait = RETRY_BACKOFF * attempt
                print(f"      [REINTENTO {attempt}/{MAX_RETRIES}] {exc}. Esperando {wait}s...")
                time.sleep(wait)

        time.sleep(DELAY_BETWEEN_FILES)

    return [
        os.path.join(species_dir, f)
        for f in os.listdir(species_dir)
        if f.endswith(".mp3")
    ]


def convert_audio_to_spectrogram(mp3_path: str) -> np.ndarray | None:
    try:
        try:
            waveform, sr = librosa.load(mp3_path, sr=16000, mono=True, duration=5)
        except Exception as load_exc:
            print(f"      librosa.load fallo: {load_exc}")
            import scipy.io.wavfile as wav_reader
            audio_segment = (
                AudioSegment.from_mp3(mp3_path)
                .set_frame_rate(16000).set_channels(1).set_sample_width(2)[:5000]
            )
            temp_wav = mp3_path.replace(".mp3", "_fallback.wav")
            audio_segment.export(temp_wav, format="wav")
            sr, samples = wav_reader.read(temp_wav)
            waveform = samples.astype(np.float32) / 32768.0
            if os.path.exists(temp_wav):
                os.remove(temp_wav)

        mel_spectrogram = librosa.feature.melspectrogram(
            y=waveform, sr=sr, n_mels=128, n_fft=2048,
            hop_length=512, fmin=50, fmax=8000,
        )
        mel_db      = librosa.power_to_db(mel_spectrogram, ref=np.max, top_db=80)
        mel_db_norm = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

        mel_db_norm_t = mel_db_norm.T
        img_pil = Image.fromarray(
            ((mel_db_norm_t - mel_db_norm_t.min()) / (mel_db_norm_t.max() - mel_db_norm_t.min() + 1e-6) * 255).astype(np.uint8),
            mode="L"
        ).resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

        rgb_array = np.stack([np.array(img_pil)] * 3, axis=-1)
        return rgb_array.astype(np.float32) / 255.0

    except Exception as exc:
        print(f"      No se pudo procesar '{os.path.basename(mp3_path)}': {exc}")
        return None


def augment_spectrogram(spectrogram: np.ndarray) -> list[np.ndarray]:
    augmented = []

    noise = np.random.normal(0, 0.015, spectrogram.shape).astype(np.float32)
    augmented.append(np.clip(spectrogram + noise, 0.0, 1.0))

    shift     = np.random.randint(10, 40)
    direction = np.random.choice([-1, 1])
    augmented.append(np.roll(spectrogram, shift * direction, axis=1))

    scale = np.random.uniform(0.80, 1.20)
    augmented.append(np.clip(spectrogram * scale, 0.0, 1.0))

    augmented.append(np.fliplr(spectrogram))

    return augmented


feature_tensors: list[np.ndarray] = []
class_labels:    list[int]        = []

for class_id, species in enumerate(SPECIES_CATALOG):
    audio_paths = download_species_recordings(species["scientific_name"], SAMPLES_PER_SPECIES)
    print(f"   Procesando espectrogramas de: {species['common_name']} ...")

    class_spectrograms = []
    for audio_path in audio_paths:
        spectrogram = convert_audio_to_spectrogram(audio_path)
        if spectrogram is not None:
            class_spectrograms.append(spectrogram)

    original_count = len(class_spectrograms)
    print(f"      {original_count} espectrogramas originales generados.")

    if 0 < original_count < AUGMENTATION_THRESHOLD:
        augmented_samples = []
        for spec in class_spectrograms:
            augmented_samples.extend(augment_spectrogram(spec))
        class_spectrograms.extend(augmented_samples)
        print(f"      Data augmentation: {original_count} -> {len(class_spectrograms)} muestras.")

    for spec in class_spectrograms:
        feature_tensors.append(spec)
        class_labels.append(class_id)

    if class_id < len(SPECIES_CATALOG) - 1:
        time.sleep(DELAY_BETWEEN_SPECIES)

X = np.array(feature_tensors)
y = np.array(class_labels)

if len(X) == 0:
    raise RuntimeError("No se pudo construir el conjunto de datos.")

print(f"\nDataset: {len(X)} muestras para {NUM_CLASSES} clases.")
for class_id, species in enumerate(SPECIES_CATALOG):
    print(f"   Clase {class_id} ({species['common_name']}): {int(np.sum(y == class_id))} muestras")

min_samples_per_class = min(np.bincount(y))
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y if min_samples_per_class >= 5 else None,
)
print(f"Division: {len(X_train)} entrenamiento / {len(X_val)} validacion.")

class_weights_array = compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train), y=y_train,
)
class_weight_dict = dict(enumerate(class_weights_array))

y_train_cat = to_categorical(y_train, NUM_CLASSES)
y_val_cat   = to_categorical(y_val,   NUM_CLASSES)

print("\nConfigurando MobileNetV2 con pesos ImageNet...")

base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

pooled_features = GlobalAveragePooling2D()(base_model.output)
dense_1         = Dense(256, activation="relu")(pooled_features)
dropout_1       = Dropout(0.4)(dense_1)
dense_2         = Dense(128, activation="relu")(dropout_1)
dropout_2       = Dropout(0.3)(dense_2)
output_logits   = Dense(NUM_CLASSES, activation="softmax")(dropout_2)

classifier_model = Model(inputs=base_model.input, outputs=output_logits)
classifier_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

print(f"FASE 1 — Entrenamiento con base congelada")

history_fase1 = classifier_model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=20,
    batch_size=8,
    class_weight=class_weight_dict,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
    ],
)

best_val_acc_fase1 = max(history_fase1.history.get("val_accuracy", [0]))
print(f"\nFASE 1 concluida. Mejor val_accuracy: {best_val_acc_fase1:.4f}")

print("\nFASE 2 — Descongelando las ultimas 30 capas de MobileNetV2...")

base_model.trainable = True
capas_a_congelar = len(base_model.layers) - 30
for i, layer in enumerate(base_model.layers):
    layer.trainable = (i >= capas_a_congelar)

classifier_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history_fase2 = classifier_model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=30,
    batch_size=8,
    class_weight=class_weight_dict,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=4, min_lr=1e-7, verbose=1),
    ],
)

best_val_acc_fase2 = max(history_fase2.history.get("val_accuracy", [0]))

MODEL_OUTPUT_PATH = "modelo_aves_real.h5"
classifier_model.save(MODEL_OUTPUT_PATH)

print(f"\nEntrenamiento concluido.")
print(f"   val_accuracy Fase 1: {best_val_acc_fase1:.4f} ({best_val_acc_fase1 * 100:.1f}%)")
print(f"   val_accuracy Fase 2: {best_val_acc_fase2:.4f} ({best_val_acc_fase2 * 100:.1f}%)")
print(f"   Modelo exportado a '{MODEL_OUTPUT_PATH}'.")
