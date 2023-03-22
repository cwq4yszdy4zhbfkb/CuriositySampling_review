from setuptools import setup

setup(
    name="curiositysampling",
    version="0.1.0",
    author="CWq4YsZDY4ZhBfKb",
    author_email="cwq4yszdy4zhbfkb@gmail.com",
    packages=[
        "curiositysampling",
        "curiositysampling.core",
        "curiositysampling.test",
        "curiositysampling.utils",
        "curiositysampling.models",
        "curiositysampling.ray",
    ],
    url="https://github.com/cwq4yszdy4zhbfkb/CuriositySampling_review",
    license="LICENSE.txt",
    description="Curiosity Sampling is a package for MD accelerated sampling usign Reinforcement Learning",
    long_description=open("README.md").read(),
    install_requires=[],
)
