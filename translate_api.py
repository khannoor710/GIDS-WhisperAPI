from flask import Flask, request, send_file, jsonify
import os
import logging
import fitz  # PyMuPDF for PDF manipulation
from transformers import MarianMTModel, MarianTokenizer
import torch

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load MarianMT model and tokenizer for French to English translation
model_name = 'Helsinki-NLP/opus-mt-fr-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def translate_text(text):
    """Translate French text to English using MarianMT model."""
    sentences = text.split('. ')
    translated_sentences = []
    for sentence in sentences:
        batch = tokenizer([sentence], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        translated = model.generate(**batch)
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        translated_sentences.append(translated_text)
    return '. '.join(translated_sentences)

def translate_pdf(input_pdf_path, output_pdf_path):
    """Translate all text in the PDF from French to English while preserving layout and ensuring text alignment."""
    doc = fitz.open(input_pdf_path)
    
    # Path to font file
    font_path = "/Users/dt226003/Whisper/GIDS-WhisperAPI/fonts/Arial/Arial-Regular.ttf"
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"Font file not found: {font_path}")
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        original_text = span["text"].strip()

                        # Skip digits and non-text elements, focus only on text to be translated
                        if original_text and not original_text.isdigit():
                            # Translate the text from French to English
                            translated_text = translate_text(original_text)

                            # Define the bounding box for the text
                            bbox = span["bbox"]

                            # Draw a white rectangle over the French text to "erase" it
                            page.draw_rect(
                                fitz.Rect(bbox),
                                color=(1, 1, 1),  # White color to match background
                                fill=True
                            )

                            # Insert the translated English text in the same location
                            page.insert_text(
                                (bbox[0], bbox[1]),  # Coordinates for the original French text
                                translated_text,      # The translated English text
                                fontsize=span["size"],  # Keep the original font size
                                fontname="helv",  # Use Helvetica (or change if you need a different font)
                                fontfile=font_path,  # Path to the custom font
                                color=(0, 0, 0),  # Black color for the text
                                overlay=True  # Overlay so that we don't affect non-text elements
                            )

    # Save the translated PDF with the layout intact
    doc.save(output_pdf_path)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload, translation, and PDF conversion."""
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

            # Step 1: Translate the PDF
            translate_pdf(input_pdf_path, output_pdf_path)

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
