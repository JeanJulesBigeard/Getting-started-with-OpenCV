from flask import Flask,render_template,request
import numpy as np
import cv2
import base64
import sys
import json


# Define App
app = Flask(__name__,template_folder="templates")

# The home page is routed to index.html inside
@app.route('/')
def index():
   return render_template('index.html')

# Load Digit Recogniztion model
net = cv2.dnn.readNetFromONNX('model.onnx')

# Implements softmax function
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# Handles uploaded image
@app.route('/upload',methods=["POST"])
def upload():
  # Get uploaded form
  d = request.form
  # Extract the data field
  data = d.get('data')
  
  # The first part of the string simply indicates 
  # what kind of file it is. So we extract only the data part. 
  data = data.split(',')[1]

  # Get base64 decoded 
  data = base64.decodebytes(data.encode())
  
  # Convert to numpy array
  nparr = np.frombuffer(data, np.uint8)

  # Read image
  img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
  cv2.imwrite("/tmp/test.jpg", img)
  
  # Create a 4D blob from image
  blob = cv2.dnn.blobFromImage(img, 1/255, (28, 28))

  # Run a model
  net.setInput(blob)
  out = net.forward()
  
  # Get a class with a highest score
  out = softmax(out.flatten())
  classId = np.argmax(out)
  confidence = out[classId]
  
  # Print results on the server side
  print("classId: {} confidence: {}".format(classId, confidence), file=sys.stdout)
  
  # Return result as a json object
  return json.dumps({'success':True, 'class': int(classId), 'confidence': float(confidence)}), 200, {'ContentType':'application/json'} 

if __name__ == '__main__':
   app.run(debug = True)
