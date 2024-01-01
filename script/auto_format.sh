#!/usr/bin/env bash
(find include -name "*.hpp" && find test -name "*.cpp") | xargs clang-format -i {}
