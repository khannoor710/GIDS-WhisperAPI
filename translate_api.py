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

def translate_text(text):
    """Translate French text to English using MarianMT model."""
    batch = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
    translated = model.generate(**batch)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def translate_pdf_page(page_content):
    """Translate an entire page of text from French to English."""
    # Join all text into a single string for the whole page
    text_to_translate = '\n'.join(page_content['texts'])
    translated_text = translate_text(text_to_translate)
    return {'page_num': page_content['page_num'], 'translated_text': translated_text, 'metadata': page_content['metadata']}

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
                block_text = ""
                block_meta = []
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if text:
                            block_text += " " + text
                            block_meta.append({'bbox': span['bbox'], 'size': span['size'], 'color': span.get('color', (0, 0, 0))})
                if block_text.strip():
                    page_texts.append(block_text.strip())
                    metadata.append(block_meta)
        all_texts.append({'page_num': page_num, 'texts': page_texts, 'metadata': metadata})
    return all_texts

def replace_text_in_pdf(doc, text_replacements):
    """Replace French text with English translations in the PDF."""
    for page_content in text_replacements:
        page = doc[page_content['page_num']]
        translated_text = page_content['translated_text']

        # Clear existing content on the page
        page.clean_contents()

        # Define text insertion box with padding to prevent overlap
        text_box = fitz.Rect(50, 50, page.rect.width - 50, page.rect.height - 50)  # Leave margins
        font_size = 10  # Smaller font size to fit more text
        font_name = "helv"  # Use Helvetica for simplicity

        # Insert translated text in a wrapped text box
        page.insert_textbox(text_box, translated_text, fontsize=font_size, fontname=font_name, color=(0, 0, 0), align=fitz.TEXT_ALIGN_LEFT)

def create_translated_pdf(input_pdf_path, output_pdf_path, translated_pages):
    """Create a translated PDF with replaced text."""
    doc = fitz.open(input_pdf_path)

    # Replace text in the PDF
    replace_text_in_pdf(doc, translated_pages)

    # Save the modified PDF
    doc.save(output_pdf_path)
    doc.close()

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
            create_translated_pdf(input_pdf_path, output_pdf_path, translated_pages)

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
