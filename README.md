## Sources helping to fix errors

### TensorFlowTTS

#### NVDIA GPG error

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

#### TensorFlow version error why installing

```
#0 37.55 ERROR: Could not find a version that satisfies the requirement tensorflow-gpu==2.7.0 (from TensorFlowTTS==0.0) (from versions: 0.12.1, 1.0.0, 1.0.1, 1.1.0, 1.2.0, 1.2.1, 1.3.0, 1.4.0, 1.4.1, 1.5.0, 1.5.1, 1.6.0, 1.7.0, 1.7.1, 1.8.0, 1.9.0, 1.10.0, 1.10.1, 1.11.0, 1.12.0, 1.12.2, 1.12.3, 1.13.1, 1.13.2, 1.14.0, 1.15.0, 1.15.2, 1.15.3, 1.15.4, 1.15.5, 2.0.0, 2.0.1, 2.0.2, 2.0.3, 2.0.4, 2.1.0, 2.1.1, 2.1.2, 2.1.3, 2.1.4, 2.2.0, 2.2.1, 2.2.2, 2.2.3, 2.3.0, 2.3.1, 2.3.2, 2.3.3, 2.3.4, 2.4.0, 2.4.1, 2.4.2, 2.4.3, 2.4.4, 2.5.0, 2.5.1, 2.5.2, 2.6.0, 2.6.1, 2.6.2, 2.12.0)
#0 37.55 ERROR: No matching distribution found for tensorflow-gpu==2.7.0 (from TensorFlowTTS==0.0)
#0 37.58 WARNING: You are using pip version 20.2.4; however, version 21.3.1 is available.
#0 37.58 You should consider upgrading via the '/usr/bin/python3 -m pip install --upgrade pip' command.
```
