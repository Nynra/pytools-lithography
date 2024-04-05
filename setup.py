import setuptools
import os


with open("README.md", "r") as fh:
    long_description = fh.read()

extras_require = {
    "ocr": [
        "easyocr"
    ],
}

install_requires=[
    "matplotlib",
    "numpy",
    "opencv-python",
    "pandas",
    "scipy",
    "pytools_image_processing @ git+https://git@github.com/Nynra/pytools-image-processing.git@0.0.3#egg=pytools_image_processing"
]


# Get the absolute path to the directory containing setup.py
here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "src/pytools_lithography/__about__.py"), "r") as f:
    about = {}
    exec(f.read(), about)


setuptools.setup(
    include_package_data=True,
    name=about["__title__"],
    version=about["__version__"],
    author=about["__author__"],
    author_email=about["__author_email__"],
    description=about["__description__"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    py_modules=[about["__title__"]],
    package_dir={"": "src"},
    install_requires=install_requires,
    extras_require=extras_require
)