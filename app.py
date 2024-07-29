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

def mask_number_in_text(text):
    def mask_number(match):
        number_str = match.group()
        number_str = number_str.replace(' ', '').replace(',', '').replace('-', '')
        if len(number_str) <= 4:
            return number_str
        return number_str[:2] + '*' * (len(number_str) - 4) + number_str[-2:]
    
    masked_text = re.sub(r'\b\d[\d ,\-]*\d\b', mask_number, text)
    
    return masked_text

def load_whisper_model(model_name="base"):
    model = whisper.load_model(model_name)
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model

def create_pdf(transcribed_text, pdf_file_path):
    doc = SimpleDocTemplate(pdf_file_path, pagesize=letter)
    flowables = []
    styles = getSampleStyleSheet()
    style = styles['BodyText']
    para = Paragraph(transcribed_text, style)
    flowables.append(para)
    doc.build(flowables)

def create_word(transcribed_text, word_file_path):
    doc = Document()
    doc.add_paragraph(transcribed_text)
    doc.save(word_file_path)

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

@app.route('/api/transcribe-live', methods=['POST'])
def transcribe_live():
    print('inside live transcribe')
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    file_path = f"./{audio_file.filename}"
    audio_file.save(file_path)
    result = model.transcribe(file_path, language="en", task="translate", fp16=False)
    transcribed_text = result.get('text', '')
    print(transcribed_text)
    masked_transcript = mask_number_in_text(transcribed_text)
    os.remove(file_path)

    return jsonify({'transcribed_text': masked_transcript})

@app.route('/api/download', methods=['POST'])
def create_protectedfile():
    data = request.json     
    transcribedText = data.get('transcribedText')
    file_type = data.get('fileType')

    if file_type == 'pdf':
        pdf_file_path = './transcribed_text.pdf'
        create_pdf(transcribedText, pdf_file_path)
        file_to_zip = pdf_file_path
    elif file_type == 'doc':
        word_file_path = './transcribed_text.docx'
        create_word(transcribedText, word_file_path)
        file_to_zip = word_file_path

    protected_zip_path = f'protected_{file_type}.zip'
    pyminizip.compress(file_to_zip, None, protected_zip_path, "yourpassword", 5)
    
    return send_file(protected_zip_path, as_attachment=True)

if __name__ == '__main__':
    app.run(port=8080)
