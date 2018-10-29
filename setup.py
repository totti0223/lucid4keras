import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lucid4keras",
    version="0.1",
    author="Yosuke Toda",
    author_email="tyosuke@aquaseerser.com",
    description="wrapper to make lucid package run with keras models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/totti0223/lucid4keras",
    packages=setuptools.find_packages(),
    keywords=[
        "tensorflow",
        "machine learning",
        "neural networks",
        "convolutional neural networks",
        "feature visualization",
        "optimization",
        "keras",
        "CNN",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    install_requires=[
    "lucid>=0.3",
    "keras>=2.0",
    ],
)