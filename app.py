from flask import Flask, request, jsonify
import torch
import whisper
import os
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


# Function to load Whisper model with GPU support if available
def load_whisper_model(model_name="base"):
    model = whisper.load_model(model_name)
    if torch.cuda.is_available():
        model = model.to("cuda")  # Move model to GPU
    return model

# Load the default Whisper model and track the model name
current_model_name = "base"
model = load_whisper_model(current_model_name)
options = whisper.DecodingOptions(language="en")

@app.route('/api/transcription', methods=['POST'])
def create_transcription():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    file_path = f"./{audio_file.filename}"
    audio_file.save(file_path)
    result = model.transcribe(file_path, language="en", task="translate",fp16=False)
    transcribed_text = result.get('text', '')
    os.remove(file_path)

    return jsonify({'transcribed_text': transcribed_text})

@app.route('/api/model', methods=['POST'])
def update_model():
    global model, current_model_name
    model_name = request.json.get('model_name', 'base')
    model = load_whisper_model(model_name)
    current_model_name = model_name
    return jsonify({'message': f'Model switched to {model_name}'}), 200

@app.route('/api/model', methods=['GET'])
def get_current_model():
    return jsonify({'current_model': current_model_name})

@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({'message': 'API is working!'})

if __name__ == '__main__':
    app.run(debug=True, port=8080)
