FROM python:3.7

RUN apt-get update
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y build-essential cmake

RUN pip install -U pip six numpy wheel setuptools mock 'future>=0.17.1'
RUN pip install -U keras_applications --no-deps
RUN pip install -U keras_preprocessing --no-deps

RUN curl https://bazel.build/bazel-release.pub.gpg | apt-key add -
RUN echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
RUN apt update && apt install bazel-2.0.0

RUN git clone -b 'r2.1' --single-branch --depth 1 https://github.com/tensorflow/tensorflow.git
WORKDIR /tensorflow
RUN ./configure
RUN bazel-2.0.0 build -c opt //tensorflow/tools/pip_package:build_pip_package
RUN ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
RUN pip install /tmp/tensorflow_pkg/tensorflow-2.1.1-cp37-cp37m-linux_x86_64.whl
WORKDIR /

RUN git clone -b 'v0.2.8-p0' --single-branch --depth 1 https://github.com/Dobiasd/FunctionalPlus && cd FunctionalPlus && mkdir -p build && cd build && cmake .. && make && make install
RUN git clone -b '3.3.7' --single-branch --depth 1 https://github.com/eigenteam/eigen-git-mirror && cd eigen-git-mirror && mkdir -p build && cd build && cmake .. && make && make install && ln -s /usr/local/include/eigen3/Eigen /usr/local/include/Eigen
RUN git clone -b 'v3.7.3' --single-branch --depth 1 https://github.com/nlohmann/json && cd json && mkdir -p build && cd build && cmake -DBUILD_TESTING=OFF .. && make && make install
RUN git clone -b 'v0.14.4-p0' --single-branch --depth 1 https://github.com/Dobiasd/frugally-deep && cd frugally-deep && mkdir -p build && cd build && cmake .. && make && make install

# To have downloaded the Keras models already
RUN CUDA_VISIBLE_DEVICES='' python3 ./frugally-deep/keras_export/save_application_examples.py

ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache

RUN rm -rf frugally-deep
RUN git clone -b 'master' --single-branch --depth 1 https://github.com/Dobiasd/frugally-deep && cd frugally-deep && mkdir -p build && cd build && cmake .. && make && make install
RUN g++ -std=c++14 -O3 -DNDEBUG -march=native frugally-deep/test/applications_performance.cpp -o applications_performance

RUN cat /proc/cpuinfo | grep 'name' | uniq
RUN g++ --version
RUN CUDA_VISIBLE_DEVICES='' PYTHONUNBUFFERED=TRUE taskset --cpu-list 1 python3 ./frugally-deep/keras_export/save_application_examples.py | grep --line-buffered -e 'on average' -e 'Conversion of model'
RUN ./applications_performance | grep --line-buffered -e 'on average' -e 'Loading, constructing, testing'
