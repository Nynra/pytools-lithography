import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    install_requires = fh.read().splitlines()

setuptools.setup(
    name="pytools_lithography",
    version="0.0.1",
    author="Nynra",
    author_email="nynradev@pm.me",
    description="Some usefull functions for lithography processes.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    py_modules=["pytools_image_lithography"],
    package_dir={"": "src"},
    install_requires=install_requires,
)