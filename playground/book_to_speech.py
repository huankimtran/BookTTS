import logging
from argparse import ArgumentParser, Namespace
from io import BytesIO, StringIO
from pathlib import Path
from typing import Callable, List

import regex as re
import spacy
import soundfile as sf
import numpy as np

from tqdm import tqdm

CPU_MAX_CHAR_COUNT = 200


def convert_pdf_to_text(pdf_converter: str, pdf_file_path: str) -> str:
    if pdf_converter.lower() == 'textract':
        import textract
        text = textract.process(pdf_file_path)
        return text.decode('utf-8')
    elif pdf_converter.lower() == 'pypdf2':
        import PyPDF2
        with open(pdf_file_path, 'rb') as pdf_file:

            in_mem_text_file = StringIO()
            
            # creating a pdf reader object
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # printing number of pages in pdf file
            print(f'File has {len(pdf_reader.pages)} page(s)')
            
            for page in pdf_reader.pages:
                in_mem_text_file.write(page.extract_text())

            return in_mem_text_file.getvalue()

    raise ValueError(f'Cannot find pdf2text converter {args.pdf_converter}')


def split_full_text_to_sentences(text: str) -> List[str]:
    nlp = spacy.load('en_core_web_sm')
    tokens = nlp(text)
    return [ f'{sent.text.strip()}' for sent in tokens.sents]


def is_valid_chunk(chunk: str) -> bool:
    # Valid chunk cannot be a single non-alphanumeric character
    if chunk.find(' ') == -1 and len(chunk) == 1 and not chunk.isalnum():
        return False
    return True


def sentence_to_chunks(sentence: str, chunk_max_size: int) -> List[str]:
    words = sentence.split()
    chunk_list = []
    current_chunk = ""
    
    for word in words:
        if len(current_chunk) + len(word) + 1 <= chunk_max_size:
            current_chunk += word + " "
        else:
            current_chunk = current_chunk.strip()
            if is_valid_chunk(current_chunk):
                chunk_list.append(current_chunk)
            current_chunk = word + " "
    
    if current_chunk:
        current_chunk = current_chunk.strip()
        if is_valid_chunk(current_chunk):
            chunk_list.append(current_chunk)
    
    return chunk_list


def split_to_chunks(text: str, sentence_cleaner: Callable[[str], str], chunk_max_size: int) -> List[str]:
    sentences = split_full_text_to_sentences(text)
    cleaned_sentences = [ sentence_cleaner(s) for s in sentences]

    chunks = []

    for sentence in cleaned_sentences:
        sentence_as_chunks = sentence_to_chunks(sentence, chunk_max_size)
        chunks.extend(sentence_as_chunks)

    return chunks


def clean_sentence(sentence) -> str:
    # Replace non-alphanumeric, non-punctuation and new line with space
    replaced_text = re.sub(r'[^a-zA-Z0-9\s\p{P}]|\n', ' ', sentence, flags=re.UNICODE)
    # Replace consecutive spaces
    replaced_text = re.sub(r'\s+', ' ', sentence, flags=re.UNICODE)
    return replaced_text


def get_text_to_speech_converter(name: str) -> Callable[[str], BytesIO]:
    """
    Given the name of the text to speech converter
    return the function that uses that speech converter to convert text to speech

    :param name: Name of the text to speech converter to return
    :return: A function that uses that speech converter to convert text to speech
    """
    if name == 'fastspeech2':

        import tensorflow as tf
        from tensorflow_tts.inference import TFAutoModel, AutoProcessor

        if tf.test.is_gpu_available():
            print(f'GPU detected, running TensorFlow with GPU: {tf.test.gpu_device_name()}')
        else:
            print(f'NO GPU FOUND, running tensorflow with CPU')

        # initialize fastspeech2 model.
        fastspeech2 = TFAutoModel.from_pretrained("tensorspeech/tts-fastspeech2-ljspeech-en")


        # initialize mb_melgan model
        mb_melgan = TFAutoModel.from_pretrained("tensorspeech/tts-mb_melgan-ljspeech-en")


        # inference
        processor = AutoProcessor.from_pretrained("tensorspeech/tts-fastspeech2-ljspeech-en")

        def fashspeech2_converter(text: str) -> BytesIO:
            
            logger.debug(repr(text))
            
            input_ids = processor.text_to_sequence(text)
            logger.debug(tf.size(input_ids))
            
            # fastspeech inference
            mel_before, mel_after, duration_outputs, _, _ = fastspeech2.inference(
                input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
                speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
                speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
                f0_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),
                energy_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),
            )

            # melgan inference
            audio_before = mb_melgan.inference(mel_before)[0, :, 0]
            audio_after = mb_melgan.inference(mel_after)[0, :, 0]

            audio_before_file_obj = BytesIO()
            audio_after_file_obj = BytesIO()

            sf.write(audio_before_file_obj, audio_before, 22050, "PCM_16", format='wav')
            sf.write(audio_after_file_obj, audio_after, 22050, "PCM_16", format='wav')

            audio_before_file_obj.seek(0)
            audio_after_file_obj.seek(0)
            audio_before_file_obj.name ='before.wav'
            audio_after_file_obj.name ='after.wav'

            return audio_after_file_obj
        
        return fashspeech2_converter
    
    raise ValueError(f'Cannot find text to speech converter named {name}')


def join_speech_chunks(speech_chunks: List[BytesIO]) -> BytesIO:
    output_file_obj = BytesIO()
    
    # Create an empty list to store the audio data
    audio_data = []

    for isc in speech_chunks:
        # Load each .wav file and retrieve the audio data
        data, sample_rate = sf.read(isc)
        audio_data.append(data)

    # Concatenate the audio data
    concatenated_data = np.concatenate(audio_data)

    # Save the concatenated audio as a new .wav file
    sf.write(output_file_obj, concatenated_data, sample_rate, format='wav')

    return output_file_obj

def convert_book_to_speech(args: Namespace):
    logger.info('Converting PDF to text...')
    converted_pdf = convert_pdf_to_text(args.pdf_converter_name, pdf_file_path=args.path_to_input_doc)

    logger.info('Splitting full text into chunks...')
    chunks = split_to_chunks(converted_pdf, chunk_max_size=args.chunk_max_size, sentence_cleaner=clean_sentence)

    logger.info('Converting each chunk to speech...')
    tts_converter = get_text_to_speech_converter(args.tts_model_name)
    chunks_as_speech = [ tts_converter(chunk) for chunk in tqdm(chunks) ]

    logger.info('Joining chunks into full speech...')
    in_mem_full_speech = join_speech_chunks(chunks_as_speech)

    logger.info('Saving down to output file...')
    Path(args.path_to_output_speech).write_bytes(in_mem_full_speech.getvalue())    

    logger.info(f'Book converted to speech saved to {args.path_to_output_speech}')


def get_args_parser() -> ArgumentParser:
    parser = ArgumentParser('BookTTS - text to speech')
    parser.add_argument(
        'path_to_input_doc',
        help="Path to the input document (*.pdf,)"
    )
    parser.add_argument(
        'path_to_output_speech', default='doc_speech.wav',
        help="Path to the output speech for input document",
    )
    parser.add_argument(
        '-pcn', '--pdf_converter_name', default='textract',
        help="Name of the pdf to text converter to use",
    )
    parser.add_argument(
        '-cms', '--chunk_max_size', default=CPU_MAX_CHAR_COUNT, type=int,
        help="The maximum length of the chunk to split to",
    )
    parser.add_argument(
        '-m', '--tts_model_name', default='fastspeech2',
        help="The name of the text to speech model to use",
    )
    parser.add_argument(
        '-v', '--verbose', default=False, action='store_true',
        help="Enable verbose mode when set",
    )
    parser.set_defaults(func=convert_book_to_speech)
    return parser


def init_logging(args: Namespace):
    if args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)

    logging.basicConfig(level=level,
                        format='[%(asctime)s][%(levelname)s] - %(module)s - %(message)s',
                        handlers=[stream_handler])

if __name__ == '__main__':
    args, remaining_args = get_args_parser().parse_known_args()

    init_logging(args)

    global logger
    logger = logging.getLogger(__name__)

    if all(h not in remaining_args for h in ('-h', '--help')):
        args.func(args)