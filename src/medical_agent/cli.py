import os
import sys
import base64
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from agent import build_medical_agent, AgentState
from openai import OpenAI

def main():
    # Load environment variables
    load_dotenv()
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set your API key in the .env file.")
        sys.exit(1)
    
    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-4")
    
    # Build the agent
    medical_agent = build_medical_agent(llm)
    
    print("Medical Agent initialized. Type 'exit' to quit.")
    print("-" * 50)
    
    # Initial state
    state = {
        "messages": [],
        "medical_context": {},  # These fields are kept for compatibility
        "reasoning": "",        # but won't be used in the simplified version
        "next": ""
    }
    
    # Function to encode image to base64
    def encode_image_to_base64(image_path):
        """Encode an image file to base64 string."""
        image_path = Path(image_path)
        if not image_path.exists():
            print(f"Error: Image file not found: {image_path}")
            return None
            
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    # Main interaction loop
    while True:
        # Get user input
        print("Enter your query or command:")
        print("- Type 'image' followed by the image path to analyze a medical image")
        print("- Type 'exit' to quit")
        user_input = input("You: ")
        
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        # Check if this is an image input
        if user_input.lower().startswith("image "):
            # Extract the image path
            image_path = user_input[6:].strip()
            
            # Encode the image
            image_base64 = encode_image_to_base64(image_path)
            
            if image_base64:
                # Create image content in the format expected by the OpenAI API
                image_content = {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                }
                
                # Add user message with image to state
                state["messages"].append({"role": "user", "content": image_content, "content_type": "image"})
                print(f"Processing image: {image_path}")
            else:
                # Skip this iteration if image encoding failed
                continue
        else:
            # Regular text input
            state["messages"].append({"role": "user", "content": user_input, "content_type": "text"})
        
        # Run the agent
        state = medical_agent.invoke(state)
        
        # Get the assistant's response
        assistant_message = next((m for m in reversed(state["messages"]) if m["role"] == "assistant"), None)
        
        if assistant_message:
            print("\nMedical Assistant:", assistant_message["content"])
            print("-" * 50)

if __name__ == "__main__":
    main()