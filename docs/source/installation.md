# Installation

The `grmpy` package can be conveniently installed from the [Python Package Index](https://pypi.python.org/pypi) (PyPI) or directly from its source files. We currently support Python 3.11+ on Linux systems.

## Python Package Index

You can install the stable version of the package the usual way.

```bash
pip install grmpy
```

## Source Files

You can download the sources directly from our [GitHub repository](https://github.com/OpenSourceEconomics/grmpy.git).

```bash
git clone https://github.com/OpenSourceEconomics/grmpy.git
```

Once you obtained a copy of the source files, installing the package in editable mode is straightforward.

```bash
pip install -e .
```

## Test Suite

Please make sure that the package is working properly by running our test suite using `pytest`.

```bash
python -c "import grmpy; grmpy.test()"
```
