import os
import json
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from dataclasses import dataclass

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

from modules.RAG.rag_system import RAGSystem, create_rag_system
from modules.summarizer import summarize_documents
from modules.llm_initializer import LLMInitializer


class AgentMode(Enum):
    """Available modes for the agent"""
    SUMMARIZATION = "summarization"
    CONVERSATION = "conversation"
    DOCUMENT_UPLOAD = "document_upload"
    AUTO_DETECT = "auto_detect"


@dataclass
class AgentState:
    """Tracks the current state of the agent"""
    mode: AgentMode = AgentMode.AUTO_DETECT
    has_documents: bool = False
    rag_system: Optional[RAGSystem] = None
    conversation_history: List[Dict[str, str]] = None
    current_index_name: Optional[str] = None
    
    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []


class IntelligentAgent:
    """
    An intelligent agent that can:
    1. Detect user intent (summarization vs conversation)
    2. Handle document uploads and create vector databases
    3. Perform RAG-based conversations
    4. Summarize documents
    """
    
    def __init__(self, default_index_name: str = "intelligent-agent-index"):
        self.llm = LLMInitializer.initialize_llm()
        self.state = AgentState()
        self.default_index_name = default_index_name
        
        # Initialize tools
        self.tools = self._create_tools()
        
        # Create agent
        self.agent = self._create_agent()
        
        # Keywords for intent detection
        self.summarization_keywords = [
            "summarize", "summary", "sum up", "brief", "overview", 
            "main points", "key points", "gist", "abstract", "tldr"
        ]
        
        self.conversation_keywords = [
            "tell me about", "explain", "what is", "how does", 
            "discuss", "chat", "talk", "question", "ask"
        ]
    
    def _create_tools(self) -> List[Tool]:
        """Create tools for the agent"""
        
        def summarize_tool(file_paths: str) -> str:
            """Summarize documents from given file paths"""
            try:
                # Parse file paths (expecting comma-separated string)
                paths = [path.strip() for path in file_paths.split(",")]
                summary = summarize_documents(paths)
                return f"Document Summary:\n{summary}"
            except Exception as e:
                return f"Error summarizing documents: {str(e)}"
        
        def conversation_tool(query: str) -> str:
            """Answer questions using RAG system"""
            try:
                if not self.state.rag_system:
                    return "No documents are currently loaded. Please upload documents first."
                
                result = self.state.rag_system.query(query)
                return result["answer"]
            except Exception as e:
                return f"Error in conversation: {str(e)}"
        
        def upload_documents_tool(file_paths: str) -> str:
            """Upload documents to vector database"""
            try:
                # Parse file paths
                paths = [path.strip() for path in file_paths.split(",")]
                
                # Create RAG system
                index_name = f"{self.default_index_name}-{len(paths)}-docs"
                self.state.rag_system = create_rag_system(paths, k=5, index_name=index_name)
                self.state.has_documents = True
                self.state.current_index_name = index_name
                
                return f"Successfully uploaded {len(paths)} documents to vector database. You can now ask questions about them."
            except Exception as e:
                return f"Error uploading documents: {str(e)}"
        
        def detect_intent_tool(user_input: str) -> str:
            """Detect user intent from input"""
            intent = self._detect_intent(user_input)
            return f"Detected intent: {intent.value}"
        
        return [
            Tool(
                name="summarize_documents",
                description="Summarize documents from file paths. Input should be comma-separated file paths.",
                func=summarize_tool
            ),
            Tool(
                name="conversation_chat",
                description="Answer questions about uploaded documents using RAG system. Input should be the user's question.",
                func=conversation_tool
            ),
            Tool(
                name="upload_documents",
                description="Upload documents to vector database for conversation. Input should be comma-separated file paths.",
                func=upload_documents_tool
            ),
            Tool(
                name="detect_intent",
                description="Detect user intent from their input. Input should be the user's message.",
                func=detect_intent_tool
            )
        ]
    
    def _create_agent(self) -> AgentExecutor:
        """Create the main agent"""
        
        # Agent prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system",
            """You are an intelligent assistant that can:
        1. Summarise documents (`summarize_documents`)
        2. Upload documents (`upload_documents`)
        3. Answer questions (`conversation_chat`)
        4. Detect user intent (`detect_intent`)

        Guidelines:
        - Always upload documents before chatting about them.
        - Be concise and helpful.

        Available tools:
        {tools}

        Tool names: {tool_names}"""),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create agent
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)
    
    def _detect_intent(self, user_input: str) -> AgentMode:
        """Detect user intent from input"""
        user_input_lower = user_input.lower()
        
        # Check for summarization keywords
        if any(keyword in user_input_lower for keyword in self.summarization_keywords):
            return AgentMode.SUMMARIZATION
        
        # Check for conversation keywords
        if any(keyword in user_input_lower for keyword in self.conversation_keywords):
            return AgentMode.CONVERSATION
        
        # Check for document upload keywords
        if any(keyword in user_input_lower for keyword in ["upload", "load", "process", "add documents"]):
            return AgentMode.DOCUMENT_UPLOAD
        
        return AgentMode.AUTO_DETECT
    
    def chat(self, user_input: str, file_paths: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Main chat interface for the agent
        
        Args:
            user_input: User's message
            file_paths: Optional file paths for document processing
            
        Returns:
            Dictionary containing response and metadata
        """
        
        # Detect intent
        detected_intent = self._detect_intent(user_input)
        
        # Prepare input for agent
        if file_paths:
            file_paths_str = ",".join(file_paths)
            agent_input = f"User message: {user_input}\nFile paths: {file_paths_str}"
        else:
            agent_input = user_input
        
        # Execute agent
        try:
            result = self.agent.invoke({"input": agent_input})
            response = result["output"]
            
            # Update conversation history
            self.state.conversation_history.append({
                "user": user_input,
                "agent": response,
                "intent": detected_intent.value,
                "timestamp": self._get_timestamp()
            })
            
            return {
                "response": response,
                "intent": detected_intent.value,
                "has_documents": self.state.has_documents,
                "conversation_history": self.state.conversation_history[-5:],  # Last 5 exchanges
                "status": "success"
            }
            
        except Exception as e:
            error_response = f"I encountered an error: {str(e)}. Please try again or rephrase your request."
            return {
                "response": error_response,
                "intent": detected_intent.value,
                "has_documents": self.state.has_documents,
                "status": "error",
                "error": str(e)
            }
    
    def direct_summarize(self, file_paths: List[str]) -> str:
        """Direct summarization without agent"""
        try:
            return summarize_documents(file_paths)
        except Exception as e:
            return f"Error in direct summarization: {str(e)}"
    
    def direct_upload_and_chat(self, file_paths: List[str], question: str) -> Dict[str, Any]:
        """Direct upload and chat without agent"""
        try:
            # Create RAG system
            index_name = f"{self.default_index_name}-direct-{len(file_paths)}"
            rag_system = create_rag_system(file_paths, k=5, index_name=index_name)
            
            # Update state
            self.state.rag_system = rag_system
            self.state.has_documents = True
            self.state.current_index_name = index_name
            
            # Query
            result = rag_system.query(question)
            
            return {
                "answer": result["answer"],
                "sources": result.get("sources", []),
                "num_sources": result.get("num_sources", 0),
                "status": "success"
            }
            
        except Exception as e:
            return {
                "answer": f"Error: {str(e)}",
                "status": "error",
                "error": str(e)
            }
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "mode": self.state.mode.value,
            "has_documents": self.state.has_documents,
            "current_index": self.state.current_index_name,
            "conversation_length": len(self.state.conversation_history),
            "available_tools": [tool.name for tool in self.tools]
        }
    
    def reset_agent(self) -> None:
        """Reset agent state"""
        self.state = AgentState()
        print("Agent state reset successfully!")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Convenience functions for quick usage
def create_intelligent_agent(index_name: str = "intelligent-agent-index") -> IntelligentAgent:
    """Create and return an intelligent agent"""
    return IntelligentAgent(default_index_name=index_name)


def quick_summarize(file_paths: List[str]) -> str:
    """Quick summarization function"""
    agent = create_intelligent_agent()
    return agent.direct_summarize(file_paths)


def quick_chat(file_paths: List[str], question: str) -> Dict[str, Any]:
    """Quick chat function"""
    agent = create_intelligent_agent()
    return agent.direct_upload_and_chat(file_paths, question)


# Example usage
if __name__ == "__main__":
    # Create agent
    agent = create_intelligent_agent()
    
    # Example file paths
    file_paths = [r"E:\langchain_summarizer_project\data\sample.txt"]
    
    print("=== Intelligent Agent Demo ===")
    
    # Test 1: Summarization
    print("\n1. Testing Summarization:")
    response1 = agent.chat("Please summarize these documents", file_paths)
    print(f"Response: {response1['response']}")
    
    # Test 2: Document Upload and Chat
    print("\n2. Testing Document Upload and Chat:")
    response2 = agent.chat("Upload these documents and tell me what they're about", file_paths)
    print(f"Response: {response2['response']}")
    
    # Test 3: Follow-up Question
    print("\n3. Testing Follow-up Question:")
    response3 = agent.chat("What are the main topics discussed?")
    print(f"Response: {response3['response']}")
    
    # Test 4: Agent Status
    print("\n4. Agent Status:")
    status = agent.get_agent_status()
    print(f"Status: {json.dumps(status, indent=2)}")