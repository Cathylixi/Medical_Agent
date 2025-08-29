import os
import base64
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from agent import build_medical_agent, AgentState
from openai import OpenAI  # Update import
from medical_agent.utils import ROOT_DIR

def main():
    # Load environment variables
    load_dotenv()
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set your API key in the .env file.")
        return
    
    # Function to encode image to base64
    def read_image(image_path):
        """Encode an image file to base64 string."""
        image_path = Path(image_path)
        if not image_path.exists():
            print(f"Error: Image file not found: {image_path}")
            return None
            
        with open(image_path, "rb") as image_file:
            return image_file.read()
    
    # Initialize the LLM with vision capabilities
    llm = ChatOpenAI(
        model="gpt-4-vision-preview",  # Updated model name
        temperature=0.7,
        max_tokens=1024
    )
    
    # Build the agent
    medical_agent = build_medical_agent()
    
    # Get image path from user
    print("Medical Image Analysis Example")
    print("-" * 50)
    image_path = os.path.join(ROOT_DIR, "../../data/test_jpg/pic1.jpg")
    
    # Encode the image
    image_file = read_image(image_path)
    
    if not image_file:
        print("Failed to process the image. Please check the path and try again.")
        return
    
    # Create image content in the format expected by the OpenAI API
    image_content = {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{image_file}"}
    }
    
    # Initial state with image
    state = {
        "messages": [{"role": "user", "content": image_content, "content_type": "image"}],
        "medical_context": {},  # These fields are kept for compatibility
        "reasoning": "",        # but won't be used in the simplified version
        "next": ""
    }
    
    # Get user's question about the image
    # question = input("What would you like to know about this medical image? ")
    question = "请提取图片中的文字信息并打印"
    
    # Add the question to the state
    state["messages"].append({"role": "user", "content": question, "content_type": "text"})
    
    print("\nProcessing your request...")
    print("-" * 50)
    
    # Run the agent
    state = medical_agent.invoke(state)
    
    # Get the assistant's response
    assistant_message = next((m for m in reversed(state["messages"]) if m["role"] == "assistant"), None)
    
    if assistant_message:
        print("\nMedical Assistant:", assistant_message["content"])
        print("-" * 50)

if __name__ == "__main__":
    main()