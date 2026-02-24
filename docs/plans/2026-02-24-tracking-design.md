# Design: Objektspårning och klassificering (08_tracking.ipynb)

## Kontext
Fristående notebook men förutsätter att elever gjort 06_keras.ipynb och 07_rps_kamera.ipynb. Skriven på svenska. Körs i Google Colab med GPU. Inga API-nycklar krävs.

## Mål
- Elever ser en färdig detektions- och tracking-pipeline köra direkt (YOLO + ByteTrack)
- Elever förstår skillnaden mellan detektion, tracking och klassificering
- Grunduppgift: elever tränar en egen klassificerare (MobileNetV2) och pluggar in den
- Utmaningsuppgift: elever provar zero-shot klassificering med CLIP utan träningsbilder

---

## Stack
| Komponent | Bibliotek | Syfte |
|-----------|-----------|-------|
| Detektion | YOLOv8 (ultralytics) | Hittar objekt i varje frame, 80 COCO-klasser |
| Tracking | ByteTrackTracker (roboflow/trackers) | Persistenta ID:n och trajektorier över frames |
| Visualisering | supervision | Ritar boxes, etiketter, trajektorier |
| Grunduppgift | MobileNetV2 (Keras) | Transfer learning på egna Drive-bilder |
| Utmaningsuppgift | CLIP (openai/clip) | Zero-shot klassificering med textbeskrivningar |

## Pipeline
```
Frame
  → YOLOv8           → bounding boxes + COCO-etikett (t.ex. "bottle")
  → ByteTrackTracker → samma objekt får samma ID över tid
  → (valfritt) CNN   → klipper ut varje box → klassificerar → visar båda etiketter
```

---

## Struktur (6 sektioner)

### 1. Intro
Markdown-cell: vad är detektion, tracking och klassificering? Enkel pipeline-illustration.
Vad YOLO redan kan (80 COCO-klasser) och varför egna klassificerare kan behövas.

### 2. Installation & laddning
```python
!pip install ultralytics trackers supervision
```
Ladda `yolov8n` (liten och snabb) och `ByteTrackTracker`. En cell, en output.

### 3. Välj input
```python
INPUT = 'video'  # Ändra till 'kamera' för live-kamera
```
- `'video'`: Filuppladdning till Colab (`files.upload()`), processas frame för frame, annoterat resultat sparas som ny videofil
- `'kamera'`: JS-webcam som i nb 07 — Python-callback tar emot frames från webbläsaren, returnerar annoterat frame

### 4. Bas-pipeline
YOLO detekterar → ByteTrack tilldelar ID:n → supervision ritar bounding boxes, YOLO-etiketter och trajektorier.
- Video: sparas till Drive
- Kamera: visas live i output-cellen

Eleverna ser tracking i aktion utan att träna något.

### 5. Grunduppgift — Transfer learning (MobileNetV2)
**Mappstruktur på Drive:**
```
My Drive/min_klassificerare/
  mugg/
  flaska/
  etui/
  ...
```
Samma mönster som nb 07 — elever lägger egna bilder i mappar.

**Träning:**
- Ladda bilder från Drive
- Finjustera MobileNetV2 (fryst baskropp, ny topp)
- Spara modell till Drive

**Plugga in:**
Pipelinen kör igen med ett extra steg:
- Klipp ut varje trackad bounding box
- Kör genom MobileNetV2
- Visa båda etiketter: `YOLO: bottle | Din: min_flaska`

### 6. Utmaningsuppgift — CLIP zero-shot
```python
kategorier = ["min mugg", "mitt etui", "min telefon"]
```
Inga träningsbilder. CLIP jämför varje boxklipp mot texterna och väljer närmaste.
Pluggar in i samma pipeline på samma sätt som MobileNetV2.

Pedagogisk poäng: visar att ML inte alltid kräver lablade dataset.

---

## Dataflöde
```
Video-frame (numpy) → YOLO → sv.Detections → ByteTrackTracker → tracked Detections
tracked Detections → för varje box: crop → resize(224×224) → MobileNetV2 / CLIP → etikett
```

## Input/output per läge
| Läge | Input | Output |
|------|-------|--------|
| Video | .mp4/.avi uppladdad till Colab | Annoterad videofil sparad till Drive |
| Kamera | JS-webcam i webbläsaren | Live-annoterat feed i output-cellen |
