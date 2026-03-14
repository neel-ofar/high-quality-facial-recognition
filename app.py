# Enhanced Face Recognition App with Ollama LLM Integration
import face_recognition
import pickle
import cv2
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import os
import requests
import json
import base64

app = Flask(__name__, static_url_path='')

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def call_ollama(prompt, model="llama2"):
    """Call Ollama LLM API"""
    try:
        url = f"{OLLAMA_HOST}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json().get('response', '')
        return None
    except Exception as e:
        print(f"[ERROR] Ollama API call failed: {str(e)}")
        return None

def generate_face_description(names, num_faces):
    """Generate intelligent description using Ollama"""
    if not names:
        return "No faces detected in the image."
    
    unique_names = list(set(names))
    unknown_count = names.count("Unknown")
    known_names = [n for n in names if n != "Unknown"]
    
    prompt = f"""You are a helpful assistant for a face recognition system. 
    
The system detected {num_faces} face(s) in the uploaded image.
Recognized faces: {', '.join(known_names) if known_names else 'None'}
Unknown faces: {unknown_count}

Please provide a brief, friendly summary (2-3 sentences) of what was detected. 
Be conversational and helpful."""
    
    description = call_ollama(prompt)
    return description if description else f"Detected {num_faces} face(s): {', '.join(unique_names)}"

@app.route('/', methods=['GET'])
def index():
    return render_template('base.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    f = request.files['image']
    
    if f.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(f.filename):
        return jsonify({'error': 'Invalid file type. Use PNG, JPG, or JPEG'}), 400
    
    try:
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        print("[INFO] Loading encodings...")
        data = pickle.loads(open('encodings.pickle', "rb").read())
        
        print("[INFO] Processing image...")
        image = cv2.imread(file_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        print("[INFO] Recognizing faces...")
        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)
        
        names = []
        
        for encoding in encodings:
            matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.5)
            name = "Unknown"
            
            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                
                name = max(counts, key=counts.get)
            
            names.append(name)
        
        # Draw rectangles and labels
        for ((top, right, bottom, left), name) in zip(boxes, names):
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        
        # Save processed image
        output_path = os.path.join(basepath, 'uploads', f'output_{secure_filename(f.filename)}')
        cv2.imwrite(output_path, image)
        
        # Generate AI description
        ai_description = generate_face_description(names, len(names))
        
        print(f"[INFO] Detected faces: {names}")
        
        return jsonify({
            'success': True,
            'faces_detected': len(names),
            'names': names,
            'unique_names': list(set(names)),
            'ai_description': ai_description,
            'output_image': f'output_{secure_filename(f.filename)}'
        })
        
    except FileNotFoundError:
        return jsonify({'error': 'Encodings file not found. Please run encode_faces.py first'}), 500
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'ollama': OLLAMA_HOST})

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    port = int(os.getenv('PORT', 8000))
    print(f"[INFO] Starting server on port {port}")
    print(f"[INFO] Ollama host: {OLLAMA_HOST}")
    http_server = WSGIServer(('0.0.0.0', port), app)
    http_server.serve_forever()
