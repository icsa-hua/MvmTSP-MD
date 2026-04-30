import setuptools 

with open("README.md", "r") as file: 
    long_description = file.read()

with open("requirements.txt", "r") as file:
    requirements = file.read().splitlines()

setuptools.setup(
    name="MV-MTSP",
    version="0.1.0",
    description="A package for simulating and solving a Mv-mTSP", 
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="", 
    packages=setuptools.find_packages(include=["dummy_app", "dummy_app.*"]),
    install_requires=requirements,
    python_requires='>=3.9'
)
