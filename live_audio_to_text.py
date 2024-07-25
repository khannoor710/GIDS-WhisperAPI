from flask import Flask, request, jsonify
import whisper
import numpy as np
import torch
from pydub import AudioSegment
import io
import re
from word2number import w2n
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the Whisper model (adjust model name as needed)
model_name = "base"  # Default to English model
audio_model = whisper.load_model(model_name)

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


@app.route('/transcribe', methods=['POST'])
def transcribe():
    global audio_model

    try:
        # Check if 'audio' file part is present in the request
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file part in the request'}), 400

        audio_file = request.files['audio']
        
        # Read the audio file using pydub
        audio = AudioSegment.from_file(io.BytesIO(audio_file.read()))
        
        # Ensure the audio is in the correct format (mono, 16kHz)
        audio = audio.set_channels(1).set_frame_rate(16000)
        
        # Convert audio to numpy array
        audio_np = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
        
        # Perform transcription using Whisper model with specified language
        result = audio_model.transcribe(audio_np, language="en", fp16=torch.cuda.is_available(),task="translate")
        transcript = result['text'].strip()

        masked_transcript = mask_number_in_text(transcript)

        print(masked_transcript)

        return jsonify({'transcript': masked_transcript})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True,port=5000)
