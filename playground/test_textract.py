import textract

def pdf_to_text(file_path):
    text = textract.process(file_path)
    return text.decode('utf-8')

# Usage
pdf_file = input('Path to pdf: ')
converted_text = pdf_to_text(pdf_file)

with open('sample.txt', mode='w') as f:
    f.write(converted_text)