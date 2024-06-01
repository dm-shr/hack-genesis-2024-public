import os
import PyPDF2
import pytesseract
from pdf2image import convert_from_path
from argparse import ArgumentParser
from tqdm import tqdm


parser = ArgumentParser()
parser.add_argument("-i", "--input", dest="input_dir",
                    help="Input directory with the PDFs")
parser.add_argument("-o", "--output", dest="output_dir",
                    help="Output directory with for the TXT files")

args = parser.parse_args()


# Read files from a directory
input_dir = args.input_dir
output_dir = args.output_dir
file_names = os.listdir(input_dir)

for file_name in tqdm([f for f in file_names if f.endswith(".pdf")]):
    pagewise_text_list = []
    pdf_document = os.path.join(input_dir, file_name)
    with open(pdf_document, "rb") as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        num_pages = len(reader.pages)

        # Iterate through the pages and extract text
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text = page.extract_text() # TODO: what happens if there is a table?

            if len(text) == 0: # let's use OCR
                images = convert_from_path(pdf_document, first_page=page_num + 1, last_page=page_num + 1)
                text = ''
                for image_num in range(len(images)):
                    text += pytesseract.image_to_string(images[image_num], lang='rus')

            pagewise_text_list.append(text)
            output_file_path = os.path.join(output_dir, file_name.replace(".pdf", ""))
            if not os.path.exists(output_file_path):
                os.makedirs(output_file_path)

            for page_num, text in enumerate(pagewise_text_list):
                page_file_path = os.path.join(output_file_path, f"page_{page_num + 1}.txt")
                with open(page_file_path, "w") as page_file:
                    page_file.write(text)
