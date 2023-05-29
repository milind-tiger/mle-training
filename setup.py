import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="housing",
    version="v0.1",
    author="Milind Thakur",
    author_email="milind.thakur@tigeranalytics.com",
    description="A small example package",
    
    url="https://github.com/pypa/sampleproject",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=[
        "requests",
        'importlib-metadata; python_version == "3.10.9"',
        "argparse",
        "numpy",
        "pandas",
        "scikit-learn",
    ],
)
