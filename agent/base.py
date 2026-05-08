"""agent/base.py"""
import os
from openai import OpenAI
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod


class Agent(ABC):
    """
    Base class for all agents.
    Provides OpenAI client initialization and common functionality.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 2048
    ):
        """
        Initialize the agent with OpenAI client.
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment variable.
            base_url: Base URL for API. If None, will use default OpenAI endpoint.
            model: Model to use for generation.
            temperature: Temperature for response generation.
            max_tokens: Maximum tokens to generate.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize OpenAI client with custom base URL if provided
        if self.base_url and self.base_url != "https://api.openai.com/v1":
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        else:
            self.client = OpenAI(api_key=self.api_key)
        
        self.conversation_history: list[Dict[str, str]] = []
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to conversation history.
        
        Args:
            role: Message role ('user', 'assistant', or 'system')
            content: Message content
        """
        self.conversation_history.append({"role": role, "content": content})
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
    
    def get_history(self) -> list[Dict[str, str]]:
        """
        Get the full conversation history.
        
        Returns:
            List of message dictionaries
        """
        return self.conversation_history.copy()
    
    def call_llm(
        self,
        system_prompt: str,
        user_message: str,
        conversation_history: Optional[list[Dict[str, str]]] = None
    ) -> str:
        """
        Call the LLM with system prompt and user message.
        
        Args:
            system_prompt: System prompt for the agent
            user_message: User's message
            conversation_history: Optional conversation history to include
            
        Returns:
            Generated response content
        """
        messages = [{"role": "system", "content": system_prompt}]
        
        if conversation_history:
            messages.extend(conversation_history)
        
        messages.append({"role": "user", "content": user_message})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            }, 
        )
        
        return response.choices[0].message.content
    
    @abstractmethod
    def process(self, input_text: str) -> str:
        """
        Process input and generate response.
        
        Args:
            input_text: Input from the other agent or user
            
        Returns:
            Agent's response
        """
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for this agent.
        
        Returns:
            System prompt string
        """
        pass
