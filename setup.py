from setuptools import setup, find_packages

# Define the Conda-specific packages to exclude
exclude_packages = ["_libgcc_mutex", "mkl", "cudatoolkit"]

# Function to filter out excluded packages
def parse_requirements(file):
    with open(file, "r") as f:
        required = f.read().splitlines()
    # Filter out the Conda-specific packages
    required = [pkg for pkg in required if not any(exclude in pkg for exclude in exclude_packages)]
    return required

required = parse_requirements("requirements.txt")

setup(
    name="afm-molecular-fingerprints",  
    version="1.0.0",  
    description="Python library to perform molecular fingerprints extraction and virtual screening from experimental HR-AFM images.",  # Description
    author="Manuel González Lastre", 
    author_email="manuel.e.g.l1999@gmail.com", 
    packages=find_packages(),  
    install_requires=required,  
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Attribution-NonCommercial 4.0 International",
        "Operating System :: OS Independent",
    ],
    license="Attribution-NonCommercial 4.0 International",
    python_requires=">=3.10",
)
