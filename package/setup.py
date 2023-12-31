# Get version
import re

name = "chromatinhd_manuscript"

import setuptools

VERSIONFILE = "src/" + name + "/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    version = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

# get long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name=name,
    version=version,
    # author="Wouter Saelens",
    # author_email="wouter.saelens@gmail.com",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DeplanckeLab/ChromatinHD_manuscript",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "chromatinhd",
        "jupyterlab",
        "jupytext",
        # "manuscript",
    ],
    extras_require={
        "full": [],
        "dev": [],
    },
)
