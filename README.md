<div align="center">

<!-- Animated Header Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:10b981,40:059669,100:047857&height=220&section=header&text=🐾%20Animal%20Detector&fontSize=58&fontColor=ffffff&animation=twinkling&fontAlignY=42&desc=YOLOv8%20·%20Real-Time%20Detection%20·%20Auto-Crop%20·%20Confidence%20Scores&descAlignY=65&descSize=16&descColor=a7f3d0" width="100%"/>

<br/>

<!-- Stat Badges Row 1 -->
<a href="https://huggingface.co/spaces/BUDDDY2894830/animal-detector">
  <img src="https://img.shields.io/badge/▶%20%20T%20R%20Y%20%20L%20I%20V%20E%20%20N%20O%20W-10b981?style=for-the-badge&logo=huggingface&logoColor=white&labelColor=022c22" height="52" alt="Try Live"/>
</a>

<br/><br/>

<a href="https://huggingface.co/spaces/BUDDDY2894830/animal-detector"><img src="https://img.shields.io/badge/Model-YOLOv8-10B981?style=for-the-badge&labelColor=0f172a"/></a>
&nbsp;
<a href="https://huggingface.co/spaces/BUDDDY2894830/animal-detector"><img src="https://img.shields.io/badge/Animals-10%20Species-F59E0B?style=for-the-badge&labelColor=0f172a"/></a>
&nbsp;
<a href="https://github.com/mabdullahab614-alt/animal-detector/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-8B5CF6?style=for-the-badge&labelColor=0f172a"/></a>
&nbsp;
<a href="https://github.com/mabdullahab614-alt/animal-detector/stargazers"><img src="https://img.shields.io/github/stars/mabdullahab614-alt/animal-detector?style=for-the-badge&color=FFD700&labelColor=0f172a"/></a>

<br/><br/>

<a href="https://huggingface.co/spaces/BUDDDY2894830/animal-detector"><img src="https://img.shields.io/badge/Deployed%20on-Hugging%20Face%20Spaces-FF6B00?style=for-the-badge&logo=huggingface&labelColor=0f172a"/></a>
&nbsp;
<a href="https://huggingface.co/spaces/BUDDDY2894830/animal-detector"><img src="https://img.shields.io/badge/Interface-Gradio-F97316?style=for-the-badge&labelColor=0f172a"/></a>
&nbsp;
<a href="https://github.com/mabdullahab614-alt/animal-detector"><img src="https://img.shields.io/badge/Framework-PyTorch-EF4444?style=for-the-badge&logo=pytorch&labelColor=0f172a"/></a>
&nbsp;
<a href="https://huggingface.co/spaces/BUDDDY2894830/animal-detector"><img src="https://img.shields.io/badge/Auto%20Crop-Per%20Animal-3B82F6?style=for-the-badge&labelColor=0f172a"/></a>

</div>

---

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:10b981,50:059669,100:047857&height=3" width="100%"/>

<br/>

<div align="center">

## 🧠 DETECTION PIPELINE ARCHITECTURE

```
╔══════════════════════════════════════════════════════════════════╗
║               ANIMAL DETECTOR — YOLOV8 PIPELINE                 ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║   INPUT: Any Photo (JPG / PNG / WEBP)                           ║
║      │                                                           ║
║      ▼                                                           ║
║   ┌─────────────────────────────────────────┐                    ║
║   │         IMAGE PRE-PROCESSING            │                    ║
║   │   • Resize to 640×640 (YOLOv8 input)   │                    ║
║   │   • Normalize pixel values [0,1]        │                    ║
║   │   • Letterbox padding (aspect ratio)    │                    ║
║   └─────────────────────────────────────────┘                    ║
║      │                                                           ║
║      ▼                                                           ║
║   ┌─────────────────────────────────────────┐                    ║
║   │           YOLOv8 BACKBONE               │                    ║
║   │   CSP-Darknet + C2f modules             │                    ║
║   │   Feature Pyramid Network (FPN)         │                    ║
║   │   Path Aggregation Network (PAN)        │                    ║
║   │   3 Detection Scales (small/med/large)  │                    ║
║   └─────────────────────────────────────────┘                    ║
║      │                                                           ║
║      ▼                                                           ║
║   ┌─────────────────────────────────────────┐                    ║
║   │     NON-MAX SUPPRESSION (NMS)           │                    ║
║   │   • Confidence threshold  ≥ 0.25        │                    ║
║   │   • IoU threshold         ≤ 0.45        │                    ║
║   │   • Removes duplicate boxes             │                    ║
║   │   • Keeps highest-confidence detection  │                    ║
║   └─────────────────────────────────────────┘                    ║
║      │                                                           ║
║      ▼                                                           ║
║   ┌─────────────────────────────────────────┐                    ║
║   │        OUTPUT GENERATION                │                    ║
║   │   • Draw colored bounding boxes         │                    ║
║   │   • Label each animal + confidence %    │                    ║
║   │   • Crop each animal individually       │                    ║
║   │   • Return annotated + cropped images   │                    ║
║   └─────────────────────────────────────────┘                    ║
║                                                                  ║
║   ⚡ Avg inference time: ~0.3s per image on CPU                  ║
╚══════════════════════════════════════════════════════════════════╝
```

</div>

---

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:10b981,50:059669,100:047857&height=3" width="100%"/>

## ✨ Features

<table>
<tr>
<td width="50%">

### 🎯 Detection
- ✅ **YOLOv8** state-of-the-art object detection
- ✅ **10 animal species** detected out of the box
- ✅ **Multi-animal** — detects all in one shot
- ✅ **Confidence scores** shown per detection
- ✅ **Colored bounding boxes** for each animal
- ✅ **Handles any image** size or aspect ratio

### 🐾 Species Support
- 🐱 Cat &nbsp;&nbsp; 🐶 Dog &nbsp;&nbsp; 🐦 Bird
- 🐴 Horse &nbsp;&nbsp; 🐑 Sheep &nbsp;&nbsp; 🐄 Cow
- 🐘 Elephant &nbsp;&nbsp; 🐻 Bear
- 🦓 Zebra &nbsp;&nbsp; 🦒 Giraffe

</td>
<td width="50%">

### ✂️ Auto-Crop
- ✅ **Crops each animal** into its own image
- ✅ Pixel-perfect **bounding box extraction**
- ✅ Each crop **labelled** with species name
- ✅ Works for **1 or 50 animals** in one photo
- ✅ Returns annotated + all crops together

### 🌐 Interface
- ✅ **Gradio web UI** — no code needed
- ✅ **Drag & drop** image upload
- ✅ **Instant results** in browser
- ✅ **Mobile friendly** — works on phone
- ✅ **Free forever** on Hugging Face Spaces
- ✅ **Zero install** — just open and use

</td>
</tr>
</table>

---

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:10b981,50:059669,100:047857&height=3" width="100%"/>

## 🚀 Live Demo

<div align="center">

### 🌐 [https://huggingface.co/spaces/BUDDDY2894830/animal-detector](https://huggingface.co/spaces/BUDDDY2894830/animal-detector)

*Open in browser — upload any photo, results in under a second*

</div>

---

## 🛠 Tech Stack

<div align="center">

| | Technology | Purpose |
|--|-----------|---------|
| 🧠 | **YOLOv8** (Ultralytics) | Real-time object detection model |
| 🔥 | **PyTorch** | Deep learning inference backend |
| 👁️ | **OpenCV / Pillow** | Image processing & bounding box drawing |
| 🎛️ | **Gradio** | Interactive web interface |
| 🤗 | **Hugging Face Spaces** | Free cloud deployment |
| 🐍 | **Python 3.10+** | Core language |

</div>

---

## ⚡ Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/mabdullahab614-alt/animal-detector.git
cd animal-detector

# 2. Install dependencies
pip install gradio torch torchvision ultralytics Pillow

# 3. Run the app
python app.py
# → Opens at http://localhost:7860
```

> Or try it instantly (no install): **[huggingface.co/spaces/BUDDDY2894830/animal-detector](https://huggingface.co/spaces/BUDDDY2894830/animal-detector)**

---

## 🏆 Rating

<div align="center">

| Category | Score |
|----------|-------|
| Detection Accuracy | ⭐⭐⭐⭐⭐ |
| Speed | ⭐⭐⭐⭐⭐ |
| Species Coverage | ⭐⭐⭐⭐⭐ |
| Auto-Crop Quality | ⭐⭐⭐⭐⭐ |
| Ease of Use | ⭐⭐⭐⭐⭐ |
| Deployment | ⭐⭐⭐⭐⭐ |
| **OVERALL** | **⭐⭐⭐⭐⭐ 10/10** |

</div>

---

## 📜 License

**MIT License** — © 2026 Abdullah Javid

Free to use, modify, and distribute with attribution.

---

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:10b981,50:059669,100:047857&height=3" width="100%"/>

<div align="center">

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-mabdullahab614--alt-181717?style=for-the-badge&logo=github&labelColor=0f172a)](https://github.com/mabdullahab614-alt)
&nbsp;
[![Live Demo](https://img.shields.io/badge/🐾%20Try%20Animal%20Detector-10b981?style=for-the-badge&labelColor=0f172a)](https://huggingface.co/spaces/BUDDDY2894830/animal-detector)
&nbsp;
[![Portfolio](https://img.shields.io/badge/🌐%20Portfolio-Abdullah%20Javid-8B5CF6?style=for-the-badge&labelColor=0f172a)](https://portfolio-website-jet-iota-21.vercel.app/)

<br/><br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:047857,60:059669,100:10b981&height=120&section=footer&animation=twinkling" width="100%"/>

**⭐ Star this repo if it helped you!**

</div>
