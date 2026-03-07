from flask import Flask, render_template, request, send_from_directory
import os
import cv2
from datetime import datetime
from ultralytics import YOLO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Load YOLO model
model = YOLO('yolov8n.pt')  # pretrained
target_classes = ['person','car','motorcycle','bicycle']  # classes to count

@app.route('/', methods=['GET', 'POST'])
def index():
    result_video = None
    counts = {}
    if request.method == 'POST':
        file = request.files.get('video')
        if file and file.filename != '':
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(upload_path)

            # Process video with YOLOv8
            result_video, counts = process_video(upload_path)

    return render_template("index.html", processed_video=result_video, counts=counts)

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)

def process_video(path):
    cap = cv2.VideoCapture(path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    out_name = f"result_{timestamp}.mp4"
    out_path = os.path.join(app.config['PROCESSED_FOLDER'], out_name)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # Counters
    counts = {cls:0 for cls in target_classes}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO detection
        results = model(frame)[0]

        # Reset counters each frame
        frame_counts = {cls:0 for cls in target_classes}

        for box in results.boxes:
            cls_id = int(box.cls)
            name = results.names.get(cls_id)
            if name in target_classes:
                frame_counts[name] += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, name, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)


        y_offset = 30
        total_text = "Counts: "
        for cls in target_classes:
            counts[cls] = counts.get(cls, 0) + frame_counts[cls]
            text = f"{cls}: {frame_counts[cls]}"
            cv2.putText(frame, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            y_offset += 30

        out.write(frame)

    cap.release()
    out.release()
    return out_name, counts

if __name__ == "__main__":
    app.run(debug=True)