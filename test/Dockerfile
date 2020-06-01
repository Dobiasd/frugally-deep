FROM python:3.7

RUN apt-get update

RUN apt-get install -y build-essential
RUN g++ --version

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y cmake
RUN cmake --version

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

RUN git clone -b 'v0.2.7-p0' --single-branch --depth 1 https://github.com/Dobiasd/FunctionalPlus && cd FunctionalPlus && mkdir -p build && cd build && cmake .. && make && make install
RUN git clone -b '3.3.7' --single-branch --depth 1 https://github.com/eigenteam/eigen-git-mirror && cd eigen-git-mirror && mkdir -p build && cd build && cmake .. && make && make install && ln -s /usr/lo$
RUN git clone -b 'v3.7.3' --single-branch --depth 1 https://github.com/nlohmann/json && cd json && mkdir -p build && cd build && cmake -DBUILD_TESTING=OFF .. && make && make install
RUN git clone -b 'v0.14.2-p0' --single-branch --depth 1 https://github.com/Dobiasd/frugally-deep && cd frugally-deep && mkdir -p build && cd build && cmake .. && make && make install

RUN CUDA_VISIBLE_DEVICES='' taskset --cpu-list 1 python3 ./frugally-deep/keras_export/save_application_examples.py | tee python_results.txt

RUN rm -rf frugally-deep
RUN git clone -b 'customconvolution' --single-branch --depth 1 https://github.com/Dobiasd/frugally-deep && cd frugally-deep && mkdir -p build && cd build && cmake .. && make && make install
RUN g++ -std=c++14 -O3 -march=native frugally-deep/test/applications_performance.cpp -o applications_performance

ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache
RUN cat python_results.txt | grep -e "average" -e "writing"
RUN ./applications_performance