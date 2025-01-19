# For performance tests on application models run the following from the main directory:
# docker build --rm --progress=plain -f test/Dockerfile .

FROM ubuntu:24.04

RUN apt-get update -y --fix-missing
ENV DEBIAN_FRONTEND=noninteractive
RUN TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get install -y build-essential apt-utils cmake python3 python3-pip git llvm-17 clang-17 apt-transport-https curl gnupg patchelf

RUN curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg
RUN mv bazel-archive-keyring.gpg /usr/share/keyrings
RUN echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
RUN apt-get update -y && apt-get install -y bazel-6.5.0
RUN ln -s /usr/bin/bazel-6.5.0 /usr/bin/bazel

RUN git clone -b 'v2.18.0' --single-branch --depth 1 https://github.com/tensorflow/tensorflow.git
WORKDIR /tensorflow
RUN ./configure
RUN bazel build -c opt --copt=-march=native --jobs 16 --local_ram_resources=HOST_RAM*.3 -c opt //tensorflow/tools/pip_package:wheel --repo_env=WHEEL_NAME=tensorflow_cpu
RUN pip install --break-system-packages bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow_cpu-2.18.0-cp312-cp312-linux_x86_64.whl
WORKDIR /

RUN git clone -b 'v0.2.24' --single-branch --depth 1 https://github.com/Dobiasd/FunctionalPlus && cd FunctionalPlus && mkdir -p build && cd build && cmake .. && make && make install
RUN git clone -b '3.4.0' --single-branch --depth 1 https://gitlab.com/libeigen/eigen.git && cd eigen && mkdir -p build && cd build && cmake .. && make && make install && ln -s /usr/local/include/eigen3/Eigen /usr/local/include/Eigen
RUN git clone -b 'v3.11.3' --single-branch --depth 1 https://github.com/nlohmann/json && cd json && mkdir -p build && cd build && cmake -DJSON_BuildTests=OFF .. && make && make install

ADD include frugally-deep/include
ADD keras_export frugally-deep/keras_export
ADD test frugally-deep/test

ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache
RUN g++ -std=c++14 -O3 -DNDEBUG -march=native frugally-deep/test/applications_performance.cpp -I frugally-deep/include -o applications_performance
RUN cat /proc/cpuinfo | grep 'name' | uniq
RUN g++ --version
RUN CUDA_VISIBLE_DEVICES='' PYTHONUNBUFFERED=TRUE taskset --cpu-list 1 python3 ./frugally-deep/keras_export/save_application_examples.py | grep --line-buffered -e 'on average' -e 'Conversion of model'
RUN ./applications_performance | grep --line-buffered -e 'on average' -e 'Loading, constructing, testing'