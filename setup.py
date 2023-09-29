from setuptools import setup

setup(
    name="boopy",
    version="0.1",
    author="Jonathan Palafoutas",
    description="for manipulation of padded arrays",
    packages=["boo", "tests"],
    install_requires=["numpy", "scipy"],
    tests_require=[
        "pytest",
    ],
)
