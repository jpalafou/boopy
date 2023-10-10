from setuptools import setup

setup(
    name="boopy",
    version="0.1",
    author="Jonathan Palafoutas",
    description="for manipulation of padded arrays",
    packages=["boo", "tests"],
    install_requires=["numpy==1.24.0", "cupy-cuda12x==12.2.0"],
    tests_require=[
        "pytest",
    ],
)
