from flask import Flask, request, jsonify
from ultralytics import YOLO
from flask_cors import CORS
import os
import cv2
import io
from werkzeug.utils import secure_filename
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/process": {"origins": "http://localhost:3000"}})

# Load the YOLO model
model = YOLO('./models/best.pt')
logger.info(f"OpenCV version: {cv2.__version__}")

# Class names and text files directory
class_names = [
    'Andaman damsel', 'Blackside hawkfish', 'Clown anemonefish', 'Fire Goby',
    'Peppered butterflyfish', 'Red lionfish', 'Saddleback clownfish',
    'Spotted garden eel', 'Titan triggerfish', 'convict surgeonfish'
]
text_files_dir = './models/name class'

def get_class_info(class_name):
    """Get additional information for the detected class."""
    text_file_path = os.path.join(text_files_dir, f"{class_name}.txt")
    if os.path.exists(text_file_path):
        with open(text_file_path, 'r') as file:
            return file.read()
    return "No additional information available."

@app.route('/process', methods=['POST'])
def process_image():
    logger.info("Request received!")

    # Check if an image file is included in the request
    if 'image' not in request.files:
        logger.error("No image file provided")
        return jsonify({'error': 'No image file provided'}), 400

    # Save the uploaded image
    image_file = request.files['image']
    image_filename = secure_filename(image_file.filename)
    image_path = f'/tmp/{image_filename}'
    image_file.save(image_path)
    logger.info(f"Image successfully received and saved at {image_path}")

    # Perform model prediction
    try:
        results = model.predict(source=image_path, conf=0.25, save=False)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'Error during prediction'}), 500

    # Process the results
    response_data = []
    processed_classes = set()
    image = cv2.imread(image_path)

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf = box.conf.item()

            if 0 <= cls_id < len(class_names):
                class_name = class_names[cls_id]

                if class_name not in processed_classes:
                    info = get_class_info(class_name)
                    response_data.append({
                        'class_name': class_name,
                        'confidence': conf,
                        'info': info
                    })
                    processed_classes.add(class_name)

    # Encode image for debugging or future use (if needed)
    _, buffer = cv2.imencode('.jpg', image)
    image_data = io.BytesIO(buffer)

    logger.info("Image processed successfully.")

    return jsonify({
        'results': response_data,
    })

if __name__ == '__main__':
    # Start the Flask app
    app.run(host='0.0.0.0', port=5002)
