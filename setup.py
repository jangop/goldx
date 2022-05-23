"""GoldX: Ground-Truth Explanations for Visual Classifiers.

See:
https://github.com/jangop/classicdata
"""

import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="goldx",
    version="0.1.0-alpha1",
    description="Ground-Truth Explanations for Visual Classifiers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jangop/goldx",
    author="Jan Philip Göpfert",
    author_email="janphilip@gopfert.eu",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: The Unlicense (Unlicense)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="machine learning",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10, <4",
    install_requires=["torch", "torchvision", "captum", "foolbox", "pillow"],
    extras_require={
        "dev": ["check-manifest", "black", "pylint"],
        "test": ["coverage", "pytest", "black", "pylint"],
    },
    project_urls={
        "Bug Reports": "https://github.com/jangop/goldx/issues",
        "Source": "https://github.com/jangop/goldx",
    },
)
