from flask import Flask, request, send_file, jsonify
import fitz  # PyMuPDF
from transformers import MarianMTModel, MarianTokenizer
import os
import logging
from multiprocessing import Pool

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load MarianMT model and tokenizer for French to English translation
model_name = 'Helsinki-NLP/opus-mt-fr-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_batch_texts(texts):
    """Batch translate French texts to English using MarianMT model."""
    batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    translated = model.generate(**batch)
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

def translate_pdf_page(page_content):
    """Translate a single page of text from French to English."""
    translated_texts = translate_batch_texts(page_content['texts'])
    return {'page_num': page_content['page_num'], 'translated_texts': translated_texts, 'metadata': page_content['metadata']}

def extract_text_from_pdf(pdf_path):
    """Extract texts and metadata from PDF."""
    doc = fitz.open(pdf_path)
    all_texts = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]
        page_texts = []
        metadata = []
        for block in blocks:
            if block["type"] == 0:  # Text block
                for line in block["lines"]:
                    for span in line["spans"]:
                        if span["text"].strip():
                            page_texts.append(span["text"])
                            metadata.append({'bbox': span['bbox'], 'size': span['size'], 'color': span.get('color', (0, 0, 0))})
        all_texts.append({'page_num': page_num, 'texts': page_texts, 'metadata': metadata})
    return all_texts

def create_translated_pdf(output_pdf_path, translated_pages):
    """Create a translated PDF using fpdf."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    for page in translated_pages:
        pdf.add_page()
        for metadata, translated_text in zip(page['metadata'], page['translated_texts']):
            pdf.set_font("Arial", size=int(metadata['size']))
            color = int(sum(metadata['color']) / len(metadata['color']) * 255) if metadata['color'] else 0
            pdf.set_text_color(color, color, color)
            pdf.set_xy(metadata['bbox'][0], metadata['bbox'][1])
            pdf.cell(0, 10, translated_text)

    pdf.output(output_pdf_path)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and translation."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.lower().endswith('.pdf'):
        input_pdf_path = os.path.join(os.getcwd(), 'input.pdf')
        output_pdf_path = os.path.join(os.getcwd(), 'translated_english_pdf.pdf')

        try:
            file.save(input_pdf_path)

            # Extract text from PDF
            extracted_data = extract_text_from_pdf(input_pdf_path)

            # Use multiprocessing for faster translation
            with Pool() as pool:
                translated_pages = pool.map(translate_pdf_page, extracted_data)

            # Create the translated PDF
            create_translated_pdf(output_pdf_path, translated_pages)

            # Clean up input file
            os.remove(input_pdf_path)

            if os.path.exists(output_pdf_path):
                response = send_file(output_pdf_path, as_attachment=True)
                os.remove(output_pdf_path)  # Clean up after sending
                return response
            else:
                return jsonify({"error": "Translation failed, file not found."}), 500
        except Exception as e:
            logging.error(f"Error in upload_file: {e}")
            return jsonify({"error": "Error during translation."}), 500

    return jsonify({"error": "Invalid file format"}), 400

if __name__ == '__main__':
    app.run(debug=True)
