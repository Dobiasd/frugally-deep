FROM ubuntu:20.04

RUN apt-get update -y --fix-missing
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y build-essential apt-utils cmake python3 python3-pip git
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10

RUN pip install -U pip
RUN pip install -U pip six numpy wheel setuptools mock 'future>=0.17.1'
RUN pip install -U keras_applications --no-deps
RUN pip install -U keras_preprocessing --no-deps

RUN apt-get -y install curl gnupg
RUN curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg
RUN mv bazel.gpg /etc/apt/trusted.gpg.d/
RUN echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
RUN apt-get update && apt-get install -y bazel-5.1.1
RUN ln -s /usr/bin/bazel-5.1.1 /usr/bin/bazel

RUN git clone -b 'v2.10.0' --single-branch --depth 1 https://github.com/tensorflow/tensorflow.git
WORKDIR /tensorflow
RUN ./configure
RUN bazel build --jobs 3 --local_ram_resources=HOST_RAM*.3 -c opt //tensorflow/tools/pip_package:build_pip_package
RUN ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
RUN pip install /tmp/tensorflow_pkg/tensorflow-2.10.0-cp38-cp38-linux_x86_64.whl
WORKDIR /

RUN git clone -b 'v0.2.18-p0' --single-branch --depth 1 https://github.com/Dobiasd/FunctionalPlus && cd FunctionalPlus && mkdir -p build && cd build && cmake .. && make && make install
RUN git clone -b '3.4.0' --single-branch --depth 1 https://gitlab.com/libeigen/eigen.git && cd eigen && mkdir -p build && cd build && cmake .. && make && make install && ln -s /usr/local/include/eigen3/Eigen /usr/local/include/Eigen
RUN git clone -b 'v3.10.5' --single-branch --depth 1 https://github.com/nlohmann/json && cd json && mkdir -p build && cd build && cmake -DJSON_BuildTests=OFF .. && make && make install
RUN git clone -b 'v0.15.19-p0' --single-branch --depth 1 https://github.com/Dobiasd/frugally-deep && cd frugally-deep && mkdir -p build && cd build && cmake .. && make && make install

# To have downloaded the Keras models already
RUN pip install numpy keras=="2.10.0" --upgrade
RUN CUDA_VISIBLE_DEVICES='' python ./frugally-deep/keras_export/save_application_examples.py

ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache

RUN rm -rf frugally-deep
RUN git clone -b 'master' --single-branch --depth 1 https://github.com/Dobiasd/frugally-deep && cd frugally-deep && mkdir -p build && cd build && cmake .. && make && make install
RUN g++ -std=c++14 -O3 -DNDEBUG -march=native frugally-deep/test/applications_performance.cpp -o applications_performance

RUN cat /proc/cpuinfo | grep 'name' | uniq
RUN g++ --version
RUN CUDA_VISIBLE_DEVICES='' PYTHONUNBUFFERED=TRUE taskset --cpu-list 1 python ./frugally-deep/keras_export/save_application_examples.py | grep --line-buffered -e 'on average' -e 'Conversion of model'
RUN ./applications_performance | grep --line-buffered -e 'on average' -e 'Loading, constructing, testing'