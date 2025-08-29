import os
import base64
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph
from typing import TypedDict, List, Dict, Any, Union, Literal
from pydantic import BaseModel, Field
from openai import OpenAI  # Update import
from agent import build_medical_agent, AgentState  # Add this import

# Load environment variables
load_dotenv()

def main():
    # Build the agent
    medical_agent = build_medical_agent()
    
    # Use text input
    initial_state = AgentState()
    
    # Run the agent
    result = medical_agent.invoke(initial_state)
    
    # Print the result
    # print(result)

if __name__ == "__main__":
    main()