from setuptools import setup, find_packages

setup(
    name="MITHEx",
    version="0.1.0",
    description="MITHEx is used to calculate heat exchanger costs and power conversion efficiency.",
    url="https://github.com/mgeschke85/MITHEx",
    author="",
    author_email="",
    license="NONE",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "pytest",
        "pint",
        "CoolProp"
    ],
    entry_points={"console_scripts": ["timechecker = MITHEx.main:main"]},
    classifiers=[
        "Development Status :: Concept",
        "Intended Audience :: Science/Research",
        "License :: NONE :: NONE",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.9",
    ],
)
