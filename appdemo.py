import cv2
import torch
from flask import Flask, render_template, Response
import numpy as np
app = Flask(__name__)

@app.route('/')
def index():
    """Video streaming ."""
    return render_template('demo.html')

def get_video_stream():
    """Gets the video stream from the webcam."""
    # Initialize the video capture object
    capture = cv2.VideoCapture(0)

    # Get the video stream
    while True:
        # Capture the frame
        ret, frame = capture.read()

        # Convert the frame to a NumPy array
        frame = frame.astype(np.float32)

        # Return the frame
        yield frame

def load_yolo_model():
    """Loads the YOLOv58 model."""
    # Load the model
    model = torch.hub.load('ultralytics/YOLOv58', 'YOLOv58s', pretrained=True)

    # Return the model
    return model

def detect_objects(frame, model):
    """Detects objects in the frame."""
    # Convert the frame to a Torch tensor
    frame = torch.from_numpy(frame).to(device)

    # Get the predictions
    predictions = model(frame)

    # Draw the bounding boxes on the frame
    for prediction in predictions:
        boxes = prediction['boxes'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()

        for box, label, score in zip(boxes, labels, scores):
            cv2.rectangle(frame, box, (255, 0, 0), 2)
            cv2.putText(frame, label + ' ' + str(score), (box[0], box[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))

    # Return the frame with the bounding boxes drawn on it
    return frame

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    # Get the video stream
    stream = get_video_stream()

    # Detect objects in the video stream
    for frame in stream:
        # Detect objects in the frame
        frame = detect_objects(frame, model)

        # Convert the frame to a JPEG image
        jpeg_image = cv2.imencode('.jpg', frame)[1].tobytes()

        # Return the JPEG image
        return Response(jpeg_image, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)