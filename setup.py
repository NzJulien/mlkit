from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="mlkit",
    version="1.0.0",
    description="Train and compare scikit-learn classifiers on any CSV dataset — CLI + library.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="NzJulien",
    url="https://github.com/NzJulien/mlkit",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.9",
    install_requires=["scikit-learn>=1.0", "pandas>=1.3", "numpy>=1.21"],
    extras_require={"dev": ["pytest>=7.0"]},
    entry_points={"console_scripts": ["mlkit=mlkit.cli:main"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
