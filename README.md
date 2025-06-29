# Shift Nth Row N Steps

<p align="center">
  <a href="https://github.com/34j/shift-nth-row-n-steps/actions/workflows/ci.yml?query=branch%3Amain">
    <img src="https://img.shields.io/github/actions/workflow/status/34j/shift-nth-row-n-steps/ci.yml?branch=main&label=CI&logo=github&style=flat-square" alt="CI Status" >
  </a>
  <a href="https://shift-nth-row-n-steps.readthedocs.io">
    <img src="https://img.shields.io/readthedocs/shift-nth-row-n-steps.svg?logo=read-the-docs&logoColor=fff&style=flat-square" alt="Documentation Status">
  </a>
  <a href="https://codecov.io/gh/34j/shift-nth-row-n-steps">
    <img src="https://img.shields.io/codecov/c/github/34j/shift-nth-row-n-steps.svg?logo=codecov&logoColor=fff&style=flat-square" alt="Test coverage percentage">
  </a>
</p>
<p align="center">
  <a href="https://github.com/astral-sh/uv">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv">
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff">
  </a>
  <a href="https://github.com/pre-commit/pre-commit">
    <img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white&style=flat-square" alt="pre-commit">
  </a>
</p>
<p align="center">
  <a href="https://pypi.org/project/shift-nth-row-n-steps/">
    <img src="https://img.shields.io/pypi/v/shift-nth-row-n-steps.svg?logo=python&logoColor=fff&style=flat-square" alt="PyPI Version">
  </a>
  <img src="https://img.shields.io/pypi/pyversions/shift-nth-row-n-steps.svg?style=flat-square&logo=python&amp;logoColor=fff" alt="Supported Python versions">
  <img src="https://img.shields.io/pypi/l/shift-nth-row-n-steps.svg?style=flat-square" alt="License">
</p>

---

**Documentation**: <a href="https://shift-nth-row-n-steps.readthedocs.io" target="_blank">https://shift-nth-row-n-steps.readthedocs.io </a>

**Source Code**: <a href="https://github.com/34j/shift-nth-row-n-steps" target="_blank">https://github.com/34j/shift-nth-row-n-steps </a>

---

Shift Nth row N steps in NumPy / PyTorch / TensorFlow / JAX

## Installation

Install this via pip (or your favourite package manager):

```shell
pip install shift-nth-row-n-steps
```

## Usage

```python
from shift_nth_row_n_steps import shift_nth_row_n_steps

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
shifted = shift_nth_row_n_steps(a)
print(shifted)
```

```text
[[1 2 3]
 [0 4 5]
 [0 0 7]]
```

`shift_nth_row_n_steps` is [array API](https://data-apis.org/array-api/latest/) compatible, which means it works with NumPy, PyTorch, JAX, and other libraries that implement the array API standard.

## Benchmark

![Benchmark](https://raw.githubusercontent.com/34j/shift-nth-row-n-steps/main/benchmark.webp)

## Algorithm

![Algorithm](https://raw.githubusercontent.com/34j/shift-nth-row-n-steps/main/docs/algo.png)

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- prettier-ignore-start -->
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- markdownlint-disable -->
<!-- markdownlint-enable -->
<!-- ALL-CONTRIBUTORS-LIST:END -->
<!-- prettier-ignore-end -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## Credits

[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-orange.json)](https://github.com/copier-org/copier)

This package was created with
[Copier](https://copier.readthedocs.io/) and the
[browniebroke/pypackage-template](https://github.com/browniebroke/pypackage-template)
project template.
