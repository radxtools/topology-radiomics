This article discusses how to build documentation for this project using sphinx.


# Generating Source Files

The following command will create the source documentation files

```
cd src
sphinx-apidoc -o docs/source topology_radiomics
```

# Building the docs

## Method 1

We can build using the setup utils.

```
python setup.py build_sphinx
```

## Method 2

We can build using the sphinx cli

```
cd docs
make html
```

# Notes

More information can be found here:

* [Python Docstring guide](https://google.github.io/styleguide/pyguide)

* [Napolean Guide](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/index.html)

* [sphinx-apidoc guide](https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html#sphinx-apidoc-manual-page)

* [Blog post on getting started with sphnix](https://samnicholls.net/2016/06/15/how-to-sphinx-readthedocs)