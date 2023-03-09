import cv2
import numpy as np
from flask import Flask, jsonify, request
from pyzbar.pyzbar import decode

app = Flask(__name__)


@app.route('/detect_barcode', methods=['POST'])
def detect_barcode():
    # Retrieve the image from form-data
    image = request.files['image'].read()

    # Convert the image data to an OpenCV image
    nparr = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert the image to grayscale and detect barcodes
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    barcodes = decode(gray)

    # Extract barcode information
    results = []
    for barcode in barcodes:
        barcode_data = barcode.data.decode("utf-8")
        barcode_type = barcode.type
        # Add barcode information to results
        results.append({
            'barcodeType': barcode_type,
            'barcodeData': barcode_data
        })

    # Return the results as JSON
    return jsonify(results)


if __name__ == '__main__':
    app.run()
