import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="psrtest", # Replace with your own username
    version="1.00",
    author="Zhou Jingzhe",
    author_email="",
    description="A python library for Predictive State Representation (PSR) modelling",
    long_description="",
    long_description_content_type="text/markdown",
    url="https://github.com/zhoujingzhe/CPSR-DRL-python-",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
