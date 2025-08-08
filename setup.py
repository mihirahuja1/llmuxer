from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="llmux",
    version="0.1.0",
    author="Your Name",
    description="Cut your LLM costs by 90% with one line of code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "requests>=2.31.0",
        "openai>=1.0.0",
        "anthropic>=0.18.0",
    ],
)