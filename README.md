# Medical Agent

A LLM-based medical assistant agent built using LangGraph and LangChain, capable of processing both text queries and medical images.

## Overview

This project implements an intelligent medical assistant that can answer health-related questions using a structured reasoning approach. The agent uses a graph-based architecture to:

1. Retrieve relevant medical knowledge
2. Apply medical reasoning
3. Generate comprehensive responses

The agent can now process both text-based medical queries and analyze medical images using the multimodal capabilities of GPT-4o-mini.

## Setup

### Prerequisites

- Python 3.9+
- OpenAI API key (with access to GPT-4o-mini or other vision-capable models)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Medical_Agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a .env file in the root directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Command Line Interface

Run the CLI interface:
```bash
python src/cli.py
```

To analyze a medical image:
```
You: image /path/to/your/medical/image.jpg
```

Then ask questions about the image:
```
You: What abnormalities do you see in this image?
```

### Image Example Script

For a dedicated image analysis experience:
```bash
python src/image_example.py
```

This script will prompt you for an image path and then ask what you'd like to know about the image.
