from flask import Flask, request, jsonify
import torch
import whisper
import os

app = Flask(__name__)

# Function to load Whisper model
def load_whisper_model(model_name="base"):
    return whisper.load_model(model_name)

# Load the default Whisper model and track the model name
current_model_name = "base"
model = load_whisper_model(current_model_name)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']

    # Save the file to a temporary location
    file_path = f"./{audio_file.filename}"
    audio_file.save(file_path)

    # Transcribe the audio file using Whisper
    result = model.transcribe(file_path)

    # Extract only the transcribed text
    transcribed_text = result.get('text', '')

    # Remove the temporary file
    os.remove(file_path)

    return jsonify({'transcribed_text': transcribed_text})

@app.route('/select_model', methods=['POST'])
def select_model():
    global model, current_model_name
    model_name = request.json.get('model_name', 'base')
    model = load_whisper_model(model_name)
    current_model_name = model_name
    return jsonify({'message': f'Model switched to {model_name}'}), 200

@app.route('/get_model', methods=['GET'])
def get_model():
    return jsonify({'current_model': current_model_name})

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'API is working!'})

if __name__ == '__main__':
    app.run(debug=True, port=8080)
