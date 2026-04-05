import gradio as gr
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np

# Load YOLOv8 (no trust prompt needed!)
model = YOLO("yolov8n.pt")

def detect_and_crop(image, history):
    if image is None:
        return "⚠️ Please upload an image!", [], None, history

    results = model(image)
    boxes   = results[0].boxes
    names   = results[0].names

    animal_classes = ['cat','dog','bird','horse','sheep','cow',
                      'elephant','bear','zebra','giraffe']

    cropped_images = []
    summary_lines  = []
    img_draw       = image.copy()
    draw           = ImageDraw.Draw(img_draw)
    colors         = ['#FF6B6B','#4ECDC4','#45B7D1','#96CEB4',
                      '#FFEAA7','#DDA0DD','#98D8C8','#F7DC6F']

    for i, box in enumerate(boxes):
        cls   = int(box.cls[0])
        label = names[cls]
        if label not in animal_classes:
            continue
        conf  = float(box.conf[0])
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        color = colors[i % len(colors)]

        draw.rectangle([x1,y1,x2,y2], outline=color, width=3)
        draw.rectangle([x1,y1,x1+len(label)*12+20,y1+25], fill=color)
        draw.text((x1+5,y1+4), f"{label} {conf:.0%}", fill="white")

        padding = 20
        cx1 = max(0, x1-padding)
        cy1 = max(0, y1-padding)
        cx2 = min(image.width,  x2+padding)
        cy2 = min(image.height, y2+padding)
        cropped_images.append(image.crop((cx1,cy1,cx2,cy2)))
        summary_lines.append(f"🐾 {label.upper()} — {conf:.1%} confidence")

    if not cropped_images:
        return "❌ No animals detected! Try a clearer photo.", [], None, history

    summary = f"✅ Found {len(cropped_images)} animal(s):\n" + "\n".join(summary_lines)
    history = (history or "") + f"\n{summary}\n" + "─"*30
    return summary, cropped_images, img_draw, history

def clear():
    return None, "Upload an image to start!", [], None, ""

with gr.Blocks(title="🐾 Animal Detector & Cropper") as app:
    gr.Markdown("""
    # 🐾 Animal Detector & Cropper
    ### Upload a photo → detects every animal → gives you individual cropped pics!
    """)

    with gr.Row():
        with gr.Column(scale=1):
            img_input  = gr.Image(label="Upload image", type="pil", height=280)
            detect_btn = gr.Button("🔍 Detect Animals!", variant="primary")
            clear_btn  = gr.Button("🗑️ Clear")
            gr.Markdown("""
---
### 🐾 Detectable Animals
🐱 Cat • 🐶 Dog • 🐦 Bird
🐴 Horse • 🐑 Sheep • 🐄 Cow
🐘 Elephant • 🐻 Bear
🦓 Zebra • 🦒 Giraffe
            """)

        with gr.Column(scale=2):
            result_text = gr.Markdown("_Upload an image to start!_")
            boxed_img   = gr.Image(label="Detected image", type="pil")
            gallery_out = gr.Gallery(label="Individual crops", columns=3, height=300)
            history_box = gr.Textbox(label="History", lines=5, interactive=False)

    history_state = gr.State("")

    detect_btn.click(fn=detect_and_crop,
                     inputs=[img_input, history_state],
                     outputs=[result_text, gallery_out, boxed_img, history_state])
    history_state.change(fn=lambda h: h, inputs=history_state, outputs=history_box)
    clear_btn.click(fn=clear,
                    outputs=[img_input, result_text, gallery_out, boxed_img, history_box])

app.launch()