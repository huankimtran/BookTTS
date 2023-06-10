# importing required modules
from argparse import Namespace
from io import StringIO
from pathlib import Path
import re
import PyPDF2
from functools import partial
import textract
  
pdf_file_path = input('Provide path to your pdf file: ')
output_file_path = input('Provide path to save converted text version of pdf file: ') or './output.txt'


def pdf_to_raw_text(args: Namespace, pdf_path: str) -> str:
    if args.pdf_converter.lower() == 'textract':
        text = textract.process(pdf_path)
        return text.decode('utf-8')
    elif args.pdf_converter.lower() == 'pypdf2':
        with open(pdf_file_path, 'rb') as pdf_file, open(output_file_path, 'w') as text_file:

            in_mem_text_file = StringIO()
            
            # creating a pdf reader object
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # printing number of pages in pdf file
            print(f'File has {len(pdf_reader.pages)} page(s)')
            
            for page in pdf_reader.pages:
                in_mem_text_file.write(page.extract_text())

            return in_mem_text_file.getvalue()

    raise ValueError(f'Cannot find pdf2text converter {args.pdf_converter}')


def clean_raw_text(text: str) -> str:
    cleaned_text = text
    for formater in [
        # Remove tab by space
        lambda x: x.replace('\t', ' '),
        # Sequence of newlines to one newline 
        partial(re.sub, r'\n+', '\n'),
        # Multiline sentence to one line sentence
        partial(re.sub, r'(\w+)( *\n+ *)(\w+)', '\g<1> \g<3>'),
        # Multiple spaces to one
        partial(re.sub, ' +', ' '),
    ]:
        cleaned_text = formater(cleaned_text)
    return cleaned_text

if not Path(pdf_file_path).exists():
    raise FileNotFoundError(f'{pdf_file_path} not exist')

with open(pdf_file_path, 'rb') as pdf_file, open(output_file_path, 'w') as text_file:

    in_mem_text_file = StringIO()
    
    # creating a pdf reader object
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    
    # printing number of pages in pdf file
    print(f'File has {len(pdf_reader.pages)} page(s)')
    
    for page in pdf_reader.pages:
        in_mem_text_file.write(page.extract_text())
    
    text_file.write(clean_raw_text(in_mem_text_file.getvalue())) 

print(f'Done converting pdf to text file at {output_file_path}')