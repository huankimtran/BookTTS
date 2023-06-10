from pathlib import Path
import spacy
nlp = spacy.load('en_core_web_sm')

text = Path(input('Path to text file (sample.txt): ') or 'sample.txt').read_text()

tokens = nlp(text)

print('\n'.join(['- ' + repr(f'{sent.text.strip()}') for sent in tokens.sents]))
