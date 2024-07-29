from flask import Flask, request, jsonify, send_file
import torch
import whisper
import pyminizip
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph
from docx import Document
import re
from word2number import w2n
import os
from flask_cors import CORS

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

# Function to generate pdf
def create_pdf(transcribed_text, pdf_file_path):
    # Create a SimpleDocTemplate
    doc = SimpleDocTemplate(pdf_file_path, pagesize=letter)
    # Create a list to hold the flowable elements
    flowables = []

    # Use styles for the text
    styles = getSampleStyleSheet()
    style = styles['BodyText']

    # Create a paragraph with the transcribed text
    para = Paragraph(transcribed_text, style)
    
    # Add the paragraph to the flowables
    flowables.append(para)
    
    # Build the PDF
    doc.build(flowables)

# Function to generate word doc
def create_word(transcribed_text, word_file_path):
    doc = Document()
    doc.add_paragraph(transcribed_text)
    doc.save(word_file_path)

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
    result = model.transcribe(file_path, language="en", task="translate", fp16=False)
    transcribed_text = result.get('text', '')
    masked_transcript = mask_number_in_text(transcribed_text)
    os.remove(file_path)

    return jsonify({'transcribed_text': masked_transcript})

@app.route('/api/download', methods=['POST'])
def create_protectedfile():
    data = request.json     
    transcribedText = data.get('transcribedText')
    file_type = data.get('fileType')  # Expecting 'text', 'doc', or 'pdf'
    print(transcribedText)
    print(file_type)

    if file_type == 'pdf':
        # Save the masked transcript to a PDF file
        pdf_file_path = './transcribed_text.pdf'
        create_pdf(transcribedText, pdf_file_path)
        file_to_zip = pdf_file_path
    elif file_type == 'doc':
        # Save the masked transcript to a Word file
        word_file_path = './transcribed_text.docx'
        create_word(transcribedText, word_file_path)
        file_to_zip = word_file_path
    else:
        # Save the masked transcript to a text file
        text_file_path = './transcribed_text.txt'
        with open(text_file_path, 'w') as file:
            file.write(transcribedText)
        file_to_zip = text_file_path

    # Create a password-protected ZIP file
    zip_file_path = './protected_transcript.zip'
    pyminizip.compress(file_to_zip, None, zip_file_path, 'admin@123#', 5)

    # Clean up the temporary files
    if os.path.exists(file_to_zip):
        os.remove(file_to_zip)

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
