# Objektspårning och klassificering — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Skapa `08_tracking.ipynb` — en svensk Colab-notebook där elever kör en färdig detektions- och tracking-pipeline (YOLOv8 + ByteTrack) och sedan pluggar in en egen klassificerare (MobileNetV2 eller CLIP).

**Architecture:** Sex notebook-sektioner. YOLOv8 detekterar objekt i varje frame, ByteTrackTracker tilldelar persistenta ID:n, supervision visualiserar bounding boxes och trajektorier. Input väljs med en konfig-cell: antingen videofil (uppladdad till Colab) eller live-kamera (JS-webcam som i nb 07). Grunduppgiften tränar MobileNetV2 på egna Drive-mappar. Utmaningsuppgiften kör CLIP zero-shot utan träningsbilder.

**Tech Stack:** ultralytics (YOLOv8), roboflow/trackers (ByteTrackTracker), supervision, TensorFlow/Keras (MobileNetV2), OpenAI CLIP, OpenCV, google.colab

---

### Task 1: Skapa notebook-fil och Section 1 — Intro (markdown)

**Files:**
- Create: `08_tracking.ipynb`

**Step 1: Skapa notebook med intro-cell**

Skapa `08_tracking.ipynb` med en enda markdown-cell:

```markdown
# Objektspårning och klassificering med AI 🎯

Du har tränat CNN:er och kört en live-kamera. Nu lägger vi till ett nytt lager: **objektspårning**.

## Tre lager i pipelinen

| Steg | Vad det gör | Vem gör det |
|------|-------------|-------------|
| **Detektion** | Hittar objekt i *ett frame* — ger bounding boxes | YOLOv8 |
| **Tracking** | Kopplar ihop boxes *över tid* — samma objekt, samma ID | ByteTrack |
| **Klassificering** | Vad exakt *är* det? — dina egna kategorier | MobileNetV2 / CLIP |

## Vad du behöver
- Ett Google-konto med Drive
- En kamera **eller** en videofil att ladda upp
- Kör alla celler uppifrån och ned

## Vad du väljer
- **Grunduppgift:** Träna en egen klassificerare (MobileNetV2) och plugga in den
- **Utmaningsuppgift:** Testa zero-shot klassificering med CLIP — inga träningsbilder krävs
```

**Step 2: Verifiera**

Öppna i Colab — markdown-cellen ska renderas korrekt med tabellen synlig.

**Step 3: Commit**

```bash
git add 08_tracking.ipynb
git commit -m "feat: add tracking notebook skeleton with intro"
```

---

### Task 2: Section 2 — Installation och imports

**Files:**
- Modify: `08_tracking.ipynb`

**Step 1: Lägg till markdown-cell**

```markdown
---
## Del 1: Installation och laddning

Kör cellen nedan för att installera nödvändiga bibliotek. Det tar ~1 minut.
```

**Step 2: Lägg till installationscell**

```python
!pip install ultralytics trackers supervision --quiet
```

**Step 3: Lägg till importcell**

```python
import os
import base64
import json
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

import supervision as sv
from ultralytics import YOLO
from trackers import ByteTrackTracker

from IPython.display import display, HTML
from google.colab import files, output as colab_output

print("Alla bibliotek importerade! ✓")
```

**Step 4: Verifiera**

Kör båda cellerna i Colab. Förväntat output:
```
Alla bibliotek importerade! ✓
```

Om `from trackers import ByteTrackTracker` misslyckas — kontrollera att rätt paketnamn används. Alternativt: `pip install roboflow-trackers` och uppdatera importen.

**Step 5: Commit**

```bash
git add 08_tracking.ipynb
git commit -m "feat: add installation and import cells"
```

---

### Task 3: Section 2 — Ladda YOLO och tracker

**Files:**
- Modify: `08_tracking.ipynb`

**Step 1: Lägg till markdown-cell**

```markdown
### Ladda detektor och tracker

YOLOv8n är den minsta och snabbaste varianten — bra för Colab.
Den är förtränad på 80 vanliga objekt (COCO-dataset): person, flaska, stol, laptop, bil...
```

**Step 2: Lägg till kodcell**

```python
# Ladda YOLOv8n — laddas ner automatiskt första gången (~6 MB)
yolo = YOLO('yolov8n.pt')

# Skapa tracker — håller reda på vilka objekt som är vilka över tid
tracker = ByteTrackTracker()

# Annotators från supervision
box_annotator   = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.6)
trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=40)

# Klassnamn som YOLO känner till
YOLO_KLASSER = yolo.model.names  # dict: {0: 'person', 1: 'bicycle', ...}

print(f"YOLO laddad ✓  ({len(YOLO_KLASSER)} klasser)")
print(f"Exempel på klasser YOLO känner igen:")
print(", ".join(list(YOLO_KLASSER.values())[:20]))
```

**Step 3: Verifiera**

Förväntat output:
```
YOLO laddad ✓  (80 klasser)
Exempel på klasser YOLO känner igen:
person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, ...
```

**Step 4: Commit**

```bash
git add 08_tracking.ipynb
git commit -m "feat: add YOLO and ByteTrack initialization cells"
```

---

### Task 4: Section 3 — Välj input

**Files:**
- Modify: `08_tracking.ipynb`

**Step 1: Lägg till markdown-cell**

```markdown
---
## Del 2: Välj input

Ändra `INPUT` nedan och kör cellen.

| Värde | Vad som händer |
|-------|----------------|
| `'video'` | Du får ladda upp en videofil (.mp4, .avi, .mov) |
| `'kamera'` | Din webcam startar direkt i nästa del |
```

**Step 2: Lägg till konfig- och upload-cell**

```python
INPUT = 'video'  # Ändra till 'kamera' för live-kamera

VIDEO_PATH  = None   # fylls i automatiskt vid upload
OUTPUT_PATH = '/content/annotated_output.mp4'

if INPUT == 'video':
    print("Ladda upp en videofil (mp4, avi, mov):")
    uploaded = files.upload()
    if uploaded:
        VIDEO_PATH = list(uploaded.keys())[0]
        print(f"Video uppladdad: {VIDEO_PATH} ✓")
    else:
        print("Ingen fil uppladdad.")
elif INPUT == 'kamera':
    print("Kamera-läge valt. Kör nästa del för att starta kameran. ✓")
else:
    raise ValueError(f"INPUT måste vara 'video' eller 'kamera', fick: '{INPUT}'")
```

**Step 3: Verifiera**

Med `INPUT = 'video'`: dialogruta för filuppladdning öppnas.
Med `INPUT = 'kamera'`: skriver ut bekräftelse utan fel.

**Step 4: Commit**

```bash
git add 08_tracking.ipynb
git commit -m "feat: add input selection cell with video upload"
```

---

### Task 5: Section 4 — Bas-pipeline för video

**Files:**
- Modify: `08_tracking.ipynb`

**Step 1: Lägg till markdown-cell**

```markdown
---
## Del 3: Kör bas-pipeline

Kör cellen nedan. YOLO detekterar objekt, ByteTrack håller koll på vilka de är över tid,
och supervision ritar ut bounding boxes med ID:n och rörelsebanor (trajektorier).

Om du kör video sparas resultatet som ny videofil. Om du kör kamera visas ett live-flöde.
```

**Step 2: Lägg till hjälpfunktion — bygg etiketter**

```python
def bygg_etikett(tracker_id, class_id, extra_label=None):
    """Bygger textetiketten som visas ovanför varje bounding box."""
    yolo_klass = YOLO_KLASSER.get(int(class_id), '?')
    etikett = f"#{tracker_id} {yolo_klass}"
    if extra_label:
        etikett += f" | {extra_label}"
    return etikett
```

**Step 3: Lägg till pipeline-funktion för video**

```python
def kör_video_pipeline(video_path, output_path, extra_classifier=None):
    """
    Kör YOLO + ByteTrack på en videofil och sparar annoterat resultat.

    Args:
        video_path:        sökväg till input-video
        output_path:       sökväg till output-video
        extra_classifier:  valfri funktion(crop_bgr) -> str, t.ex. MobileNetV2 eller CLIP
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Kunde inte öppna video: {video_path}")

    fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tot  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w, h)
    )

    # Återställ trackern för varje ny video
    global tracker
    tracker = ByteTrackTracker()

    frame_nr = 0
    print(f"Bearbetar {tot} frames ({w}×{h} @ {fps:.1f} fps)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detektion
        results    = yolo(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Tracking
        tracked = tracker.update(detections)

        # Bygg etiketter
        etiketter = []
        for tid, cid, box in zip(tracked.tracker_id, tracked.class_id, tracked.xyxy):
            extra = None
            if extra_classifier is not None:
                x1, y1, x2, y2 = map(int, box)
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    extra = extra_classifier(crop)
            etiketter.append(bygg_etikett(tid, cid, extra))

        # Rita ut
        frame = trace_annotator.annotate(frame, tracked)
        frame = box_annotator.annotate(frame, tracked)
        frame = label_annotator.annotate(frame, tracked, labels=etiketter)

        out.write(frame)
        frame_nr += 1
        if frame_nr % 30 == 0:
            print(f"  Frame {frame_nr}/{tot}...")

    cap.release()
    out.release()
    print(f"Klar! Sparad till: {output_path}")
```

**Step 4: Lägg till körningscell + visning**

```python
if INPUT == 'video' and VIDEO_PATH:
    kör_video_pipeline(VIDEO_PATH, OUTPUT_PATH)

    # Visa annoterad video direkt i notebook
    with open(OUTPUT_PATH, 'rb') as f:
        video_b64 = base64.b64encode(f.read()).decode()
    display(HTML(f'''
        <video controls width="640">
            <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
        </video>
    '''))
elif INPUT == 'kamera':
    print("Du kör i kamera-läge — hoppa till kamera-cellen nedan.")
```

**Step 5: Verifiera**

Ladda upp en kort testfilm (~10 sek). Förväntat resultat:
- Annoterad video spelas upp i notebook-cellen
- Bounding boxes med `#ID yolo_klass` ovanför
- Gröna trajektorie-linjer som visar rörelsebanor
- Inga fel i konsolen

**Step 6: Commit**

```bash
git add 08_tracking.ipynb
git commit -m "feat: add video pipeline with YOLO tracking and visualization"
```

---

### Task 6: Section 4 — Bas-pipeline för live-kamera

**Files:**
- Modify: `08_tracking.ipynb`

**Step 1: Lägg till markdown-cell**

```markdown
### Live-kamera

Kör cellen nedan om du valt `INPUT = 'kamera'`. En live-feed öppnas med bounding boxes och tracking-ID:n i realtid.
```

**Step 2: Lägg till kamera-hjälpfunktion**

```python
def starta_kamera_pipeline(extra_classifier=None):
    """
    Startar live-kamera med YOLO + ByteTrack i Colab.
    Varje frame annoteras och visas som uppdaterad bild i output-cellen.

    Args:
        extra_classifier: valfri funktion(crop_bgr) -> str
    """
    # Återställ tracker
    global tracker
    tracker = ByteTrackTracker()

    def behandla_frame(img_b64):
        """Python-callback som tar ett base64-frame och returnerar annoterat frame."""
        binary    = base64.b64decode(img_b64.split(',')[1])
        img_array = np.frombuffer(binary, dtype=np.uint8)
        frame     = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame is None:
            return ""

        results    = yolo(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        tracked    = tracker.update(detections)

        etiketter = []
        for tid, cid, box in zip(tracked.tracker_id, tracked.class_id, tracked.xyxy):
            extra = None
            if extra_classifier is not None:
                x1, y1, x2, y2 = map(int, box)
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    extra = extra_classifier(crop)
            etiketter.append(bygg_etikett(tid, cid, extra))

        frame = trace_annotator.annotate(frame, tracked)
        frame = box_annotator.annotate(frame, tracked)
        frame = label_annotator.annotate(frame, tracked, labels=etiketter)

        _, buf   = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return base64.b64encode(buf).decode('utf-8')

    colab_output.register_callback('behandla_frame', behandla_frame)

    html = """
<div style="font-family: sans-serif;">
  <video id="cam-video" width="640" height="480" autoplay playsinline
         style="display:none;"></video>
  <img id="cam-output" width="640" style="border: 2px solid #555; border-radius:6px;" />
  <div id="cam-status" style="margin-top:8px; color:#555;">Startar kamera...</div>
</div>

<script>
(async () => {
  const video   = document.getElementById('cam-video');
  const imgEl   = document.getElementById('cam-output');
  const status  = document.getElementById('cam-status');

  const stream  = await navigator.mediaDevices.getUserMedia({video: {width:640, height:480}});
  video.srcObject = stream;
  await video.play();

  const canvas  = document.createElement('canvas');
  canvas.width  = 640;
  canvas.height = 480;

  let running = true;
  status.textContent = 'Kameran är igång! ✓';

  async function loop() {
    while (running) {
      canvas.getContext('2d').drawImage(video, 0, 0, 640, 480);
      const imgData = canvas.toDataURL('image/jpeg', 0.7);

      try {
        const result = await google.colab.kernel.invokeFunction(
          'behandla_frame', [imgData], {}
        );
        const b64 = result.data['application/json'];
        if (b64) {
          imgEl.src = 'data:image/jpeg;base64,' + b64;
        }
      } catch(e) {
        status.textContent = 'Fel: ' + e.message;
        running = false;
      }
    }
  }

  loop();
})();
</script>
"""
    display(HTML(html))
    print("Kameran är igång! ✓")
```

**Step 3: Lägg till körningscell**

```python
if INPUT == 'kamera':
    starta_kamera_pipeline()
elif INPUT == 'video':
    print("Du kör i video-läge — kamera-cellen hoppas över.")
```

**Step 4: Verifiera**

Med `INPUT = 'kamera'`:
- Live-feed visas med bounding boxes runt igenkännbara objekt
- Etiketter visar `#1 person`, `#2 bottle` etc. med persistenta ID:n
- Trajektorier ritas ut när objekt rör sig

**Step 5: Commit**

```bash
git add 08_tracking.ipynb
git commit -m "feat: add live camera pipeline with YOLO tracking"
```

---

### Task 7: Section 5 — Träna MobileNetV2 (grunduppgift)

**Files:**
- Modify: `08_tracking.ipynb`

**Step 1: Lägg till markdown-cell**

```markdown
---
## Del 4: Grunduppgift — Träna din egna klassificerare

YOLO känner till 80 klasser. Men vad händer om du vill skilja på *din* mugg och *din* flaska?
Eller klassificera saker YOLO aldrig sett?

Skapa mappar i Google Drive, lägg dit egna bilder, och träna MobileNetV2 på dem.

### Mappstruktur
```
My Drive/
  min_klassificerare/
    mugg/          ← minst 20 bilder
    flaska/        ← minst 20 bilder
    telefon/       ← minst 20 bilder
    (lägg till fler mappar = fler klasser)
```

**Hur lägger man till bilder?**
1. Gå till [drive.google.com](https://drive.google.com)
2. Skapa mappen `min_klassificerare/` och undermappar för varje klass
3. Ladda upp foton (JPG, PNG — alla håll och belysningar, minst 20 per klass)
4. Kör sedan cellerna nedan
```

**Step 2: Lägg till importcell för träning**

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

from google.colab import drive
drive.mount('/content/drive')

DRIVE_ROOT       = '/content/drive/My Drive'
KLASSIFICERARE_DIR = f'{DRIVE_ROOT}/min_klassificerare'
MODELL_PATH      = f'{DRIVE_ROOT}/min_klassificerare_modell.keras'

print(f"Drive kopplat! ✓")
```

**Step 3: Lägg till bildladdningscell**

```python
IMG_SIZE = 224  # MobileNetV2 förväntar sig 224×224

def ladda_egna_bilder(data_dir):
    """
    Laddar bilder från data_dir/<klass>/ mappar.
    Varje undermapp = en klass.
    Returnerar X, y och klassnamn.
    """
    data_dir = Path(data_dir)
    klasser  = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])

    if not klasser:
        raise ValueError(f"Inga undermappar hittades i {data_dir}")

    print(f"Hittade {len(klasser)} klasser: {klasser}")

    X, y = [], []
    for i, klass in enumerate(klasser):
        klass_dir = data_dir / klass
        filer = list(klass_dir.glob('*.jpg'))  + \
                list(klass_dir.glob('*.jpeg')) + \
                list(klass_dir.glob('*.png'))  + \
                list(klass_dir.glob('*.webp'))

        laddade = 0
        for fil in filer:
            try:
                img = Image.open(fil).convert('RGB')
                img = img.resize((IMG_SIZE, IMG_SIZE))
                X.append(np.array(img, dtype='float32') / 255.0)
                y.append(i)
                laddade += 1
            except Exception as e:
                print(f"  Varning: {fil.name}: {e}")

        print(f"  {klass}: {laddade} bilder")

    return np.array(X), np.array(y), klasser

X, y, MINA_KLASSER = ladda_egna_bilder(KLASSIFICERARE_DIR)
print(f"\nX.shape: {X.shape}")
print(f"Klasser: {MINA_KLASSER}")
```

**Step 4: Lägg till träningscell**

```python
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Träning: {len(X_train)}  Validering: {len(X_val)}")

# Bygg modell med transfer learning
bas = keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
bas.trainable = False  # Frys basmodellen — vi tränar bara toppen

modell = keras.Sequential([
    bas,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(MINA_KLASSER), activation='softmax')
])

modell.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"Modell byggd ✓  ({len(MINA_KLASSER)} klasser)")
modell.summary()
```

**Step 5: Lägg till fit-cell**

```python
history = modell.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=16,
    verbose=1
)

_, val_acc = modell.evaluate(X_val, y_val, verbose=0)
print(f"\nValiderings-accuracy: {val_acc*100:.1f}%")

# Spara till Drive
modell.save(MODELL_PATH)
print(f"Modell sparad: {MODELL_PATH} ✓")
```

**Step 6: Verifiera**

Kör med ~20+ bilder per klass. Förväntat:
- Val accuracy >80% efter 10 epochs (med tillräckliga bilder)
- Modell sparad till Drive utan fel

**Step 7: Commit**

```bash
git add 08_tracking.ipynb
git commit -m "feat: add MobileNetV2 transfer learning training cells"
```

---

### Task 8: Section 5 — Plugga in MobileNetV2 i pipeline

**Files:**
- Modify: `08_tracking.ipynb`

**Step 1: Lägg till markdown-cell**

```markdown
### Plugga in klassificeraren

Nu kombinerar vi! YOLO hittar och trackar objekt. Vår MobileNetV2 berättar vad *du* kallar dem.
Båda etiketter visas: `#1 bottle | din_flaska`
```

**Step 2: Lägg till klassificerare-funktion**

```python
def mobilenet_klassificerare(crop_bgr):
    """
    Tar en bounding box-klipp (BGR numpy) och returnerar klassnamnet
    enligt den tränade MobileNetV2-modellen.
    """
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_pil = Image.fromarray(crop_rgb).resize((IMG_SIZE, IMG_SIZE))
    crop_arr = np.expand_dims(np.array(crop_pil, dtype='float32') / 255.0, axis=0)
    pred     = modell.predict(crop_arr, verbose=0)[0]
    return MINA_KLASSER[np.argmax(pred)]

print("Klassificerare redo! ✓")
print(f"Klasser: {MINA_KLASSER}")
```

**Step 3: Lägg till körningscell**

```python
# Kör pipelinen igen med din klassificerare inkopplad

if INPUT == 'video' and VIDEO_PATH:
    OUTPUT_MN = '/content/annotated_mobilenet.mp4'
    kör_video_pipeline(VIDEO_PATH, OUTPUT_MN, extra_classifier=mobilenet_klassificerare)

    with open(OUTPUT_MN, 'rb') as f:
        video_b64 = base64.b64encode(f.read()).decode()
    display(HTML(f'''
        <video controls width="640">
            <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
        </video>
    '''))

elif INPUT == 'kamera':
    starta_kamera_pipeline(extra_classifier=mobilenet_klassificerare)
```

**Step 4: Verifiera**

Etiketter ska nu visa `#1 bottle | min_flaska` (YOLO-klass + din klass sida vid sida).

**Step 5: Commit**

```bash
git add 08_tracking.ipynb
git commit -m "feat: plug MobileNetV2 classifier into tracking pipeline"
```

---

### Task 9: Section 6 — CLIP zero-shot setup (utmaningsuppgift)

**Files:**
- Modify: `08_tracking.ipynb`

**Step 1: Lägg till markdown-cell**

```markdown
---
## Del 5: Utmaningsuppgift — Zero-shot klassificering med CLIP

Vad händer om du inte har några träningsbilder alls?

**CLIP** (Contrastive Language–Image Pretraining) från OpenAI kan jämföra bilder mot *textbeskrivningar*.
Du skriver vad du letar efter — CLIP matchar.

```python
mina_kategorier = ["en röd mugg", "en vattenflaska", "ett tangentbord"]
```

Inga bilder att ladda upp. Ingen träning. Bara text.
```

**Step 2: Lägg till installationscell**

```python
!pip install git+https://github.com/openai/CLIP.git --quiet
```

**Step 3: Lägg till CLIP-setup-cell**

```python
import torch
import clip
from PIL import Image as PILImage

# Välj device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Kör CLIP på: {DEVICE}")

# Ladda CLIP-modell
clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
print("CLIP laddad! ✓")
```

**Step 4: Lägg till kategorikonfig-cell**

```python
# ── Ändra listan nedan till dina egna kategorier ──────────────────────────
CLIP_KATEGORIER = [
    "en kopp eller mugg",
    "en vattenflaska",
    "ett tangentbord",
    "en mobiltelefon",
    "en bok",
]
# ──────────────────────────────────────────────────────────────────────────

# Tokenisera kategoritexterna (görs en gång)
clip_text_tokens = clip.tokenize(CLIP_KATEGORIER).to(DEVICE)

print(f"Kategorier redo: {CLIP_KATEGORIER}")
```

**Step 5: Lägg till CLIP-klassificerare-funktion**

```python
def clip_klassificerare(crop_bgr):
    """
    Tar en bounding box-klipp (BGR numpy) och returnerar närmaste
    CLIP-kategori ur CLIP_KATEGORIER.
    """
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img  = PILImage.fromarray(crop_rgb)
    img_t    = clip_preprocess(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        img_feat  = clip_model.encode_image(img_t)
        text_feat = clip_model.encode_text(clip_text_tokens)
        sim       = (img_feat @ text_feat.T).softmax(dim=-1)

    bästa_idx = int(sim.argmax())
    return CLIP_KATEGORIER[bästa_idx]

print("CLIP-klassificerare redo! ✓")
```

**Step 6: Verifiera**

Testa manuellt innan du kör pipelinen:
```python
# Snabbtest — ladda en testbild
test_img_bgr = cv2.imread('/content/drive/My Drive/min_klassificerare/mugg/001.jpg')
resultat = clip_klassificerare(test_img_bgr)
print(f"CLIP säger: {resultat}")
```

**Step 7: Commit**

```bash
git add 08_tracking.ipynb
git commit -m "feat: add CLIP zero-shot classifier setup"
```

---

### Task 10: Section 6 — Plugga in CLIP i pipeline

**Files:**
- Modify: `08_tracking.ipynb`

**Step 1: Lägg till markdown-cell**

```markdown
### Kör pipeline med CLIP

Samma pipeline — men nu är det CLIP som klassificerar, utan en enda träningsbild.
```

**Step 2: Lägg till körningscell**

```python
if INPUT == 'video' and VIDEO_PATH:
    OUTPUT_CLIP = '/content/annotated_clip.mp4'
    kör_video_pipeline(VIDEO_PATH, OUTPUT_CLIP, extra_classifier=clip_klassificerare)

    with open(OUTPUT_CLIP, 'rb') as f:
        video_b64 = base64.b64encode(f.read()).decode()
    display(HTML(f'''
        <video controls width="640">
            <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
        </video>
    '''))

elif INPUT == 'kamera':
    starta_kamera_pipeline(extra_classifier=clip_klassificerare)
```

**Step 3: Lägg till avslutande markdown**

```markdown
---
## Bra jobbat! 🎉

Du har nu byggt en komplett AI-pipeline:

- **YOLOv8** detekterar objekt i varje frame
- **ByteTrack** håller koll på vad som är vad över tid
- **MobileNetV2** (grunduppgift) klassificerar med dina egna bilder
- **CLIP** (utmaningsuppgift) klassificerar med bara text — inga bilder alls

### Vad kan du göra härnäst?
- Testa med en film från YouTube (ladda ner som mp4)
- Lägg till fler klasser i din MobileNetV2-träning
- Ändra `CLIP_KATEGORIER` till vad du vill hitta
- Kombinera: kör MobileNetV2 och CLIP parallellt och jämför resultaten
```

**Step 4: Verifiera hela notebook**

Kör alla celler från topp till botten i Colab med `INPUT = 'video'` och en testfilm:
- [ ] Alla celler körs utan fel
- [ ] Bas-pipeline visar bounding boxes med `#ID klass`
- [ ] Trajektorier ritas ut vid rörelse
- [ ] MobileNetV2-pipeline visar `#ID yolo_klass | min_klass`
- [ ] CLIP-pipeline visar `#ID yolo_klass | clip_kategori`
- [ ] Kör om med `INPUT = 'kamera'` och verifiera live-feed

**Step 5: Final commit**

```bash
git add 08_tracking.ipynb
git commit -m "feat: complete tracking notebook — CLIP integration and final markdown"
```
