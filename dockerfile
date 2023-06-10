FROM tensorflow/tensorflow:2.7.0-gpu
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update
RUN apt-get install -y zsh tmux wget git libsndfile1 libenchant-dev
RUN pip install ipython cmake PyEnchant PyPDF2 textract regex nltk pysbd
RUN pip install git+https://github.com/huankimtran/TensorflowTTS.git
RUN pip install git+https://github.com/huankimtran/german_transliterate.git#egg=german_transliterate
# Seem like this must be installed after the  TensorflowTTS otherwise we will see import error for dataclass_transform
RUN pip install spacy && python -m spacy download en_core_web_sm
RUN python -c "import nltk;nltk.download('punkt')"
RUN mkdir /workspace
WORKDIR /workspace
