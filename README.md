# 当前大卫组维护的MQBench,逐步迭代更新功能中。
------------
[![Documentation Status](https://readthedocs.org/projects/mqbench/badge/?version=latest)](https://mqbench.readthedocs.io/en/latest/?badge=latest)
[![Lint and test](https://github.com/ModelTC/MQBench/actions/workflows/lint-and-test.yml/badge.svg?branch=main)](https://github.com/ModelTC/MQBench/actions/workflows/lint-and-test.yml)
[![license](https://img.shields.io/github/license/ModelTC/MQBench)](https://github.com/ModelTC/MQBench/blob/main/LICENSE)

## Introduction

MQBench is an open-source model quantization toolkit based on PyTorch fx.

The envision of MQBench is to provide:

- **SOTA Algorithms**. With MQBench, the hardware vendors and researchers can benefit from the latest research progress in academic.
- **Powerful Toolkits**. With the toolkit, quantization node can be inserted to the original PyTorch module automatically with respect to the specific hardware. After training, the quantized model can be smoothly converted to the format that can inference on the real device.

## Installation

```shell
git clone git@github.com:ModelTC/MQBench.git
cd MQBench
python setup.py install
```

## Documentation

MQBench aims to support (1) various deployable quantization algorithms and (2) hardware backend libraries to facilitate the development of the community.

For the detailed information, please refer to [MQBench documentation](https://mqbench.readthedocs.io/en/latest/).

## License
This project is released under the [Apache 2.0 license](LICENSE).
