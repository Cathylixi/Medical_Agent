from setuptools import setup, find_packages

setup(
    name="medical_agent",
    version="0.1.0",
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    install_requires=[
        "langchain>=0.0.335",
        "langchain-openai>=0.0.2",
        "langgraph>=0.0.15",
        "python-dotenv>=1.0.0",
        "pydantic>=2.5.0",
        "Pillow>=9.0.0",
        "openai>=1.0.0"
    ],
    author="Zhengyu Chen",
    author_email="chenzhengyu14thss@163.com",
    description="A medical assistant agent using LLMs for text and image analysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/medical-agent",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)