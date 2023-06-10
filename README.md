## Sources helping to fix errors

### TensorFlowTTS

#### Error - NVDIA GPG error

```
#0 6.804 W: GPG error: https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  InRelease: The following signatures couldn't be verified because the public key is not available: NO_PUBKEY A4B469963BF863CC
#0 6.804 E: The repository 'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  InRelease' is no longer signed.
```

According to [this](https://github.com/NVIDIA/nvidia-docker/issues/1632#issuecomment-1112667716)

These lines

```
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
```

Were added to the TensorFlowTTS dockerfile before the ```apt-get update``` line

#### Error - TensorFlow version error while installing

```
#0 37.55 ERROR: Could not find a version that satisfies the requirement tensorflow-gpu==2.7.0 (from TensorFlowTTS==0.0) (from versions: 0.12.1, 1.0.0, 1.0.1, 1.1.0, 1.2.0, 1.2.1, 1.3.0, 1.4.0, 1.4.1, 1.5.0, 1.5.1, 1.6.0, 1.7.0, 1.7.1, 1.8.0, 1.9.0, 1.10.0, 1.10.1, 1.11.0, 1.12.0, 1.12.2, 1.12.3, 1.13.1, 1.13.2, 1.14.0, 1.15.0, 1.15.2, 1.15.3, 1.15.4, 1.15.5, 2.0.0, 2.0.1, 2.0.2, 2.0.3, 2.0.4, 2.1.0, 2.1.1, 2.1.2, 2.1.3, 2.1.4, 2.2.0, 2.2.1, 2.2.2, 2.2.3, 2.3.0, 2.3.1, 2.3.2, 2.3.3, 2.3.4, 2.4.0, 2.4.1, 2.4.2, 2.4.3, 2.4.4, 2.5.0, 2.5.1, 2.5.2, 2.6.0, 2.6.1, 2.6.2, 2.12.0)
#0 37.55 ERROR: No matching distribution found for tensorflow-gpu==2.7.0 (from TensorFlowTTS==0.0)
#0 37.58 WARNING: You are using pip version 20.2.4; however, version 21.3.1 is available.
#0 37.58 You should consider upgrading via the '/usr/bin/python3 -m pip install --upgrade pip' command.
```

Change the base image of the docker file to the missing version, in this case
```
FROM tensorflow/tensorflow:2.7.0-gpu
```

#### Error - pyopenjtalk cmake missing
```
#0 35.07 Collecting pyopenjtalk
#0 35.09   Downloading pyopenjtalk-0.3.0.tar.gz (1.5 MB)
#0 35.21   Installing build dependencies: started
#0 38.03   Installing build dependencies: finished with status 'done'
#0 38.03   Getting requirements to build wheel: started
#0 38.20   Getting requirements to build wheel: finished with status 'error'
#0 38.20   ERROR: Command errored out with exit status 1:
#0 38.20    command: /usr/bin/python3 /usr/local/lib/python3.8/dist-packages/pip/_vendor/pep517/_in_process.py get_requires_for_build_wheel /tmp/tmpmilnrpp_
#0 38.20        cwd: /tmp/pip-install-vtmibszk/pyopenjtalk
#0 38.20   Complete output (23 lines):
#0 38.20   Traceback (most recent call last):
#0 38.20     File "/usr/local/lib/python3.8/dist-packages/pip/_vendor/pep517/_in_process.py", line 280, in <module>
#0 38.20       main()
#0 38.20     File "/usr/local/lib/python3.8/dist-packages/pip/_vendor/pep517/_in_process.py", line 263, in main
#0 38.20       json_out['return_val'] = hook(**hook_input['kwargs'])
#0 38.20     File "/usr/local/lib/python3.8/dist-packages/pip/_vendor/pep517/_in_process.py", line 114, in get_requires_for_build_wheel
#0 38.20       return hook(config_settings)
#0 38.20     File "/usr/local/lib/python3.8/dist-packages/setuptools/build_meta.py", line 162, in get_requires_for_build_wheel
#0 38.20       return self._get_build_requires(
#0 38.20     File "/usr/local/lib/python3.8/dist-packages/setuptools/build_meta.py", line 143, in _get_build_requires
#0 38.20       self.run_setup()
#0 38.20     File "/usr/local/lib/python3.8/dist-packages/setuptools/build_meta.py", line 267, in run_setup
#0 38.20       super(_BuildMetaLegacyBackend,
#0 38.20     File "/usr/local/lib/python3.8/dist-packages/setuptools/build_meta.py", line 158, in run_setup
#0 38.20       exec(compile(code, __file__, 'exec'), locals())
#0 38.20     File "setup.py", line 153, in <module>
#0 38.20     File "/usr/lib/python3.8/subprocess.py", line 493, in run
#0 38.20       with Popen(*popenargs, **kwargs) as process:
#0 38.20     File "/usr/lib/python3.8/subprocess.py", line 858, in __init__
#0 38.20       self._execute_child(args, executable, preexec_fn, close_fds,
#0 38.20     File "/usr/lib/python3.8/subprocess.py", line 1704, in _execute_child
#0 38.20       raise child_exception_type(errno_num, err_msg, err_filename)
#0 38.20   FileNotFoundError: [Errno 2] No such file or directory: 'cmake'
```

Installing the cmake package before installing the rest 

```
RUN pip install ipython && \
    pip install cmake && \
    pip install git+https://github.com/TensorSpeech/TensorflowTTS.git && \
    pip install git+https://github.com/repodiac/german_transliterate.git#egg=german_transliterate
```

##### Error - Docker missing runtime nvidia

```
huantran@Hagrid:~/Desktop/Working/BookTTS$ sudo docker compose up
[+] Running 1/0
 ✔ Network booktts_default            Created                                                                                                             0.1s 
 ⠋ Container booktts-tensorflowtts-1  Creating                                                                                                            0.0s 
Error response from daemon: unknown or invalid runtime name: nvidia
```

Need to install NVDIA Container Toolkit
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit

Basically, run
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

Then 
```
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
# Configure the Docker daemon to recognize the NVIDIA Container Runtime:

sudo nvidia-ctk runtime configure --runtime=docker
# Restart the Docker daemon to complete the installation after setting the default runtime:

sudo systemctl restart docker
```

#### Error - Cannot import name dataclass_transform
```
huantran@Hagrid:~/Desktop/Working$ sudo docker exec -it 60d26bc96d3a /bin/bash

________                               _______________                
___  __/__________________________________  ____/__  /________      __
__  /  _  _ \_  __ \_  ___/  __ \_  ___/_  /_   __  /_  __ \_ | /| / /
_  /   /  __/  / / /(__  )/ /_/ /  /   _  __/   _  / / /_/ /_ |/ |/ / 
/_/    \___//_/ /_//____/ \____//_/    /_/      /_/  \____/____/|__/


WARNING: You are running this container as root, which can cause new files in
mounted volumes to be created as the root user on your host machine.

To avoid this, run the container by specifying your user's userid:

$ docker run -u $(id -u):$(id -g) args...

root@60d26bc96d3a:/workspace# ls
README.md  designs  docker-compose.yaml  dockerfile  playground
root@60d26bc96d3a:/workspace# python3
Python 3.8.10 (default, Sep 28 2021, 16:10:42) 
[GCC 9.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> exit()
root@60d26bc96d3a:/workspace# cd playground/
root@60d26bc96d3a:/workspace/playground# ls
test.py
root@60d26bc96d3a:/workspace/playground# python3 test.py 
Traceback (most recent call last):
  File "test.py", line 7, in <module>
    from tensorflow_tts.inference import TFAutoModel
  File "/usr/local/lib/python3.8/dist-packages/tensorflow_tts/inference/__init__.py", line 1, in <module>
    from tensorflow_tts.inference.auto_model import TFAutoModel
  File "/usr/local/lib/python3.8/dist-packages/tensorflow_tts/inference/auto_model.py", line 24, in <module>
    from tensorflow_tts.configs import (
  File "/usr/local/lib/python3.8/dist-packages/tensorflow_tts/configs/__init__.py", line 1, in <module>
    from tensorflow_tts.configs.base_config import BaseConfig
  File "/usr/local/lib/python3.8/dist-packages/tensorflow_tts/configs/base_config.py", line 21, in <module>
    from tensorflow_tts.utils.utils import CONFIG_FILE_NAME
  File "/usr/local/lib/python3.8/dist-packages/tensorflow_tts/utils/__init__.py", line 1, in <module>
    from tensorflow_tts.utils.cleaners import (
  File "/usr/local/lib/python3.8/dist-packages/tensorflow_tts/utils/cleaners.py", line 25, in <module>
    from tensorflow_tts.utils.number_norm import normalize_numbers
  File "/usr/local/lib/python3.8/dist-packages/tensorflow_tts/utils/number_norm.py", line 26, in <module>
    import inflect
  File "/usr/local/lib/python3.8/dist-packages/inflect/__init__.py", line 76, in <module>
    from pydantic import Field, validate_arguments
  File "pydantic/__init__.py", line 2, in init pydantic.__init__
  File "pydantic/dataclasses.py", line 41, in init pydantic.dataclasses
```

Install the package named "spacy", so the docker compose should have an additional line
```
RUN pip install spacy
```

#### Error - tensorflow.python.framework.errors_impl.AlreadyExistsError: Another metric with the same name already exists.
```

    from tensorflow_tts.inference import TFAutoModel, AutoProcessor
  File "/usr/local/lib/python3.8/dist-packages/tensorflow_tts/inference/__init__.py", line 1, in <module>
    from tensorflow_tts.inference.auto_model import TFAutoModel
  File "/usr/local/lib/python3.8/dist-packages/tensorflow_tts/inference/auto_model.py", line 24, in <module>
    from tensorflow_tts.configs import (
  File "/usr/local/lib/python3.8/dist-packages/tensorflow_tts/configs/__init__.py", line 1, in <module>
    from tensorflow_tts.configs.base_config import BaseConfig
  File "/usr/local/lib/python3.8/dist-packages/tensorflow_tts/configs/base_config.py", line 21, in <module>
    from tensorflow_tts.utils.utils import CONFIG_FILE_NAME
  File "/usr/local/lib/python3.8/dist-packages/tensorflow_tts/utils/__init__.py", line 11, in <module>
    from tensorflow_tts.utils.decoder import dynamic_decode
  File "/usr/local/lib/python3.8/dist-packages/tensorflow_tts/utils/decoder.py", line 19, in <module>
    from tensorflow_addons.seq2seq import Decoder
  File "/usr/local/lib/python3.8/dist-packages/tensorflow_addons/__init__.py", line 23, in <module>
    from tensorflow_addons import activations
  File "/usr/local/lib/python3.8/dist-packages/tensorflow_addons/activations/__init__.py", line 17, in <module>
    from tensorflow_addons.activations.gelu import gelu
  File "/usr/local/lib/python3.8/dist-packages/tensorflow_addons/activations/gelu.py", line 19, in <module>
    from tensorflow_addons.utils.types import TensorLike
  File "/usr/local/lib/python3.8/dist-packages/tensorflow_addons/utils/types.py", line 25, in <module>
    from keras.engine import keras_tensor
  File "/usr/local/lib/python3.8/dist-packages/keras/__init__.py", line 25, in <module>
    from keras import models
  File "/usr/local/lib/python3.8/dist-packages/keras/models.py", line 20, in <module>
    from keras import metrics as metrics_module
  File "/usr/local/lib/python3.8/dist-packages/keras/metrics.py", line 26, in <module>
    from keras import activations
  File "/usr/local/lib/python3.8/dist-packages/keras/activations.py", line 20, in <module>
    from keras.layers import advanced_activations
  File "/usr/local/lib/python3.8/dist-packages/keras/layers/__init__.py", line 23, in <module>
    from keras.engine.input_layer import Input
  File "/usr/local/lib/python3.8/dist-packages/keras/engine/input_layer.py", line 21, in <module>
    from keras.engine import base_layer
  File "/usr/local/lib/python3.8/dist-packages/keras/engine/base_layer.py", line 43, in <module>
    from keras.mixed_precision import loss_scale_optimizer
  File "/usr/local/lib/python3.8/dist-packages/keras/mixed_precision/loss_scale_optimizer.py", line 18, in <module>
    from keras import optimizers
  File "/usr/local/lib/python3.8/dist-packages/keras/optimizers.py", line 26, in <module>
    from keras.optimizer_v2 import adadelta as adadelta_v2
  File "/usr/local/lib/python3.8/dist-packages/keras/optimizer_v2/adadelta.py", line 22, in <module>
    from keras.optimizer_v2 import optimizer_v2
  File "/usr/local/lib/python3.8/dist-packages/keras/optimizer_v2/optimizer_v2.py", line 36, in <module>
    keras_optimizers_gauge = tf.__internal__.monitoring.BoolGauge(
  File "/usr/local/lib/python3.8/dist-packages/tensorflow/python/eager/monitoring.py", line 360, in __init__
    super(BoolGauge, self).__init__('BoolGauge', _bool_gauge_methods,
  File "/usr/local/lib/python3.8/dist-packages/tensorflow/python/eager/monitoring.py", line 135, in __init__
    self._metric = self._metric_methods[self._label_length].create(*args)
tensorflow.python.framework.errors_impl.AlreadyExistsError: Another metric with the same name already exists.
```

This happened when I use the copy of the project in pip at version 1.8
This can be resolved by installing directly from TensorFlowTTS using

```
pip install git+https://github.com/TensorSpeech/TensorflowTTS.git
```

However, to freeze the repo content, I have forked the TensorflowTTS to my git account so now the below will be used

```
pip install git+https://github.com/huankimtran/TensorflowTTS.git
```

#### Error - PyEnchant missing C library
Try installing with
```
apt-get install -y libenchant-dev
```

#### Using CPU to do the inference
- On large chunk of text, the inference would usually would fail. Breaking them down into senteces and feed to the network one by one would be better. This happens with both GPU and CPU inference
- The model seems to have hight tolerance for odd words or text. It worked pretty well even when the words are stitched together like this sentence below

```
Ten years earlier,
they had entered the same building on the ground floor after my sisterwasdiagnosedwithleukemia at age three
```

- Seems like the limit for CPU is 351 chracters and 69 words
```
They are the cab, not
the gym.
THE TWO-MINUTE RULE
Even when you know you should start small, it’s easy to start too big.
When you dream about making a change, excitement inevitably takes
over and you end up trying to do too much too soon. The most effective
way I know to counteract this tendency is to use the
 Two-Minute Rule
They are the cab, not the gym.
```

#### FastSpeech2 does not seem to be able to process string with weird character standing alone
The string
```
'*'
```

will cause error
```
Traceback (most recent call last):
  File "book_to_speech.py", line 218, in <module>
    args.func(args)
  File "book_to_speech.py", line 177, in convert_book_to_speech
    chunks_as_speech = [ tts_converter(chunk) for chunk in tqdm(chunks) ]
  File "book_to_speech.py", line 177, in <listcomp>
    chunks_as_speech = [ tts_converter(chunk) for chunk in tqdm(chunks) ]
  File "book_to_speech.py", line 133, in fashspeech2_converter
    audio_before = mb_melgan.inference(mel_before)[0, :, 0]
  File "/usr/local/lib/python3.8/dist-packages/tensorflow/python/util/traceback_utils.py", line 153, in error_handler
    raise e.with_traceback(filtered_tb) from None
  File "/usr/local/lib/python3.8/dist-packages/tensorflow/python/eager/execute.py", line 58, in quick_execute
    tensors = pywrap_tfe.TFE_Py_Execute(ctx._handle, device_name, op_name,
tensorflow.python.framework.errors_impl.InvalidArgumentError:  paddings must be less than the dimension size: 3, 3 not less than 2
         [[node sequential_4/first_reflect_padding/MirrorPad
 (defined at /usr/local/lib/python3.8/dist-packages/tensorflow_tts/models/melgan.py:58)
]] [Op:__inference_inference_7343]

Errors may have originated from an input operation.
Input Source operations connected to node sequential_4/first_reflect_padding/MirrorPad:
In[0] mels (defined at /usr/local/lib/python3.8/dist-packages/tensorflow_tts/models/mb_melgan.py:174)
In[1] sequential_4/first_reflect_padding/MirrorPad/paddings:
```

### Spacy
#### Cannot load model
- Need to download the model by running the command below first

```
python -m spacy download <model_name>
```

For example

```
python -m spacy download en_core_web_sm
```