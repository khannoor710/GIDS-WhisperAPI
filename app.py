from flask import Flask, request, jsonify,send_file
import torch
import whisper
import pyminizip
import io
import re
from word2number import w2n
import os
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Function to mask and remove spaces and - 
def mask_number_in_text(text):
    def mask_number(match):
        number_str = match.group()
        # Remove any white spaces, commas, and hyphens in the number
        number_str = number_str.replace(' ', '').replace(',', '').replace('-', '')
        if len(number_str) <= 4:
            return number_str  # If the number is too short, return it as is
        return number_str[:2] + '*' * (len(number_str) - 4) + number_str[-2:]
    
    # Use regular expression to find all integer numbers in the text
    # This regex handles numbers with optional internal spaces, commas, and hyphens
    masked_text = re.sub(r'\b\d[\d ,\-]*\d\b', mask_number, text)
    
    return masked_text

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
    masked_transcript = mask_number_in_text(transcribed_text)
    os.remove(file_path)

    return jsonify({'transcribed_text': masked_transcript})

@app.route('/api/download', methods=['POST'])
def create_protectedfile():
    data = request.json     
    transcribedText = data.get('transcribedText')
    print(transcribedText)
    # Save the masked transcript to a text file
    text_file_path = './transcribed_text.txt'
    with open(text_file_path, 'w') as file:
        file.write(transcribedText)

    # Create a password-protected ZIP file
    zip_file_path = './protected_transcript.zip'
    pyminizip.compress(text_file_path, None, zip_file_path, 'admin@123#', 5)

   # Send the ZIP file to the client
    return send_file(zip_file_path, as_attachment=True, download_name='protected_transcript.zip')

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
