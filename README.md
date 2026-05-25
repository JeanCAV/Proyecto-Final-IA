# BirdScope

Aplicación hibrida para identificación de aves a partir de audio, usando una CNN basada en MobileNetV2 y espectrogramas de Mel como representación de entrada.

---

## Descripción general

BirdScope permite grabar el canto de un ave y obtener una predicción de la especie identificada, el nivel de confianza de la predicción y una grabación de referencia obtenida desde la API de [Xeno-canto](https://xeno-canto.org). Cada análisis queda registrado en un historial.

El sistema está compuesto por dos módulos:

- `cnn.py` — entrenamiento del modelo (se ejecuta una sola vez para generar `modelo_aves_real.h5`).  
- `main.py` — servidor Flask y gestión de la interfaz web.

---

## Especies reconocidas

| ID | Nombre común | Nombre científico |
| :---- | :---- | :---- |
| 0 | Bichofue Gritón | *Pitangus sulphuratus* |
| 1 | Siriri Común | *Tyrannus melancholicus* |
| 2 | Azulejo de Palma | *Thraupis palmarum* |
| 3 | Azulejo Común | *Thraupis episcopus* |
| 4 | Copetón Andino | *Zonotrichia capensis* |
| 5 | Mirla Embarradora | *Turdus ignobilis* |
| 6 | Cucarachero Común | *Troglodytes aedon* |
| 7 | Torcaza Naguiblanca | *Zenaida auriculata* |
| 8 | Colibrí Amazilia | *Amazilia tzacatl* |
| 9 | Garza del Ganado | *Bubulcus ibis* |

---

## Estructura del proyecto

```
birdscope/
├── main.py                  # Servidor Flask + pipeline de inferencia
├── cnn.py                   # Entrenamiento del modelo CNN
├── modelo_aves_real.h5      # Modelo entrenado (generado por cnn.py)
├── historial.json           # Historial de análisis (generado automáticamente)
├── templates/
│   └── index.html           # Interfaz web (Jinja2)
└── static/
    └── uploads/             # Audios y espectrogramas generados
```

---

## Requisitos

- Python 3.10 o superior  
- FFmpeg instalado en el sistema (requerido por PyDub para decodificar audio)

### Dependencias Python

```
flask
tensorflow
librosa
numpy
matplotlib
pillow
pydub
requests
scikit-learn
scipy
```

```shell
pip install -r requirements.txt
```

---

## Uso

### 1\. Entrenar el modelo

Este paso descarga grabaciones desde Xeno-canto, genera los espectrogramas y entrena la CNN. Solo es necesario ejecutarlo una vez.

```shell
python cnn.py
```

El proceso aplica transfer learning en dos fases sobre MobileNetV2:

- **Fase 1** — base convolucional congelada, 20 épocas máximo.  
- **Fase 2** — últimas 30 capas descongeladas, 30 épocas máximo.

Al finalizar se exporta `modelo_aves_real.h5` en el directorio raíz.

Si una especie tiene menos de 35 muestras disponibles, se aplica data augmentation automáticamente   
2\. Iniciar el servidor

```shell
python main.py
```

El servidor queda disponible en `http://localhost:5000`.

### 3\. Montar el servidor con ngrok (opcional)

Flask por defecto solo es accesible desde `localhost`. Si se necesita acceder a la aplicación desde otro dispositivo en la misma red o desde internet, se puede usar [ngrok](https://ngrok.com) para generar una URL pública que apunte al servidor local.

```shell
ngrok http 5000
```

### 4\. Clasificar un canto

1. Abrir `http://localhost:5000` en el navegador, o la URL de ngrok si se accede desde otro dispositivo.  
2. Pulsar **Iniciar Grabación** y acercar el micrófono al ave.  
3. Pulsar **Detener y Analizar**.  
4. El panel de resultado muestra la especie identificada, el nivel de confianza, el espectrograma y una grabación de referencia de Xeno-canto.

---

## Proceso de Predicción

```
Audio del micrófono (WebM/WAV)
        │
        ▼
  Carga con librosa (16 kHz, mono, 5 s)
  └─ fallback: PyDub + scipy si librosa falla
        │
        ▼
  Espectrograma de Mel
  (128 bins, FFT 2048, hop 512, 50–8000 Hz)
        │
        ▼
  Escala logarítmica (dB) + normalización z-score
        │
        ▼
  Redimensión a 224×224 px → tensor RGB (3 canales)
        │
        ▼
  MobileNetV2 → softmax (10 clases) → argmax
        │
        ▼
  Especie + confianza + referencia Xeno-canto
```

---

**Respuesta (JSON):**

```json
{
  "fecha": "24/05/2026 15:30:00",
  "archivo": "/static/uploads/audio_20260524_153000.wav",
  "resultado": "Copetón Andino",
  "nombre_cientifico": "Zonotrichia capensis",
  "confianza": "94.7%",
  "xeno_canto": [
    {
      "id": "123456",
      "country": "Colombia",
      "locality": "Pereira, Risaralda",
      "recorder": "J. Gómez",
      "audio_url": "https://xeno-canto.org/sounds/uploaded/..."
    }
  ]
}
```

---

## Parámetros de entrenamiento

| Parámetro | Valor |
| :---- | :---- |
| Arquitectura base | MobileNetV2 (ImageNet) |
| Tamaño de entrada | 224 × 224 × 3 |
| Muestras por especie | 50 |
| División train/val | 80% / 20% |
| Umbral augmentation | \< 35 muestras por clase |

