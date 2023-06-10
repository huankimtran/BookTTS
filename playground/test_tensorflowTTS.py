from pathlib import Path
import numpy as np
import soundfile as sf
import tensorflow as tf


from tensorflow_tts.inference import TFAutoModel, AutoProcessor

DEFAULT_SAMPLE_FILE_NAME = 'sample.txt'

text_file_path = input(f'Proivde path to the text file to convert to speech (default={DEFAULT_SAMPLE_FILE_NAME}): ') or f'./{DEFAULT_SAMPLE_FILE_NAME}'

text_file = Path(text_file_path)

if text_file.exists():
    with open(text_file_path) as f:
        text = f.read()
        print(text)
else:
    text = "Recent research at Harvard has shown meditating for as little as 8 weeks, can actually increase the grey matter in the parts of the brain responsible for emotional regulation, and learning."

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

input_ids = processor.text_to_sequence(text)
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

# save to file
sf.write('./audio_before.wav', audio_before, 22050, "PCM_16")
sf.write('./audio_after.wav', audio_after, 22050, "PCM_16")