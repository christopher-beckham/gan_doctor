#FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel
FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

ARG PIP_EXTRA_INDEX_URL

RUN pip install scipy==1.3.3
RUN pip install sklearn scikit-image eai-shuriken pyyaml h5py
WORKDIR /src
RUN mkdir /src/results && chmod 777 /src/results
COPY . .

# TensorFlow
RUN pip install tensorflow-gpu==1.15.0
# Have to do some hacky shit here so that it recognises
# the cuda libs.

RUN ln -s /usr/lib/x86_64-linux-gnu/libcublas.so.10 /usr/lib/x86_64-linux-gnu/libcublas.so.10.0
RUN ln -s /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcufft.so.10 /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcufft.so.10.0
RUN ln -s /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcurand.so.10 /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcurand.so.10.0
RUN ln -s /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcusolver.so.10 /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcusolver.so.10.0
RUN ln -s /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcusparse.so.10 /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcusparse.so.10.0
# not correct, but works??
RUN ln -s /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcudart.so.10.1 /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcudart.so.10.0

RUN export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/:/usr/local/cuda-10.1/targets/x86_64-linux/lib/:${LD_LIBRARY_PATH}"
