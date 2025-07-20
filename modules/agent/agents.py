from dataclasses import dataclass
from enum import Enum
import json
from typing import Any, Dict, List, Optional

from modules.RAG.rag_system import RAGSystem, create_rag_system
from modules.llm_initializer import LLMInitializer
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_tool_calling_agent

from modules.summarizer import summarize_documents
from config.prompts import agent_prompt


class AgentMode(Enum):
    SUMMARIZATION = "summarization"
    CONVERSATION = "conversation"
    DOCUMENT_UPLOAD = "document_upload"
    AUTO_DETECT = "auto_detect"
    
@dataclass
class AgentState:
    
    mode: AgentMode = AgentMode.AUTO_DETECT
    has_documents: bool = False
    rag_system: Optional[RAGSystem]= None
    converstation_history: List[Dict[str,str]] = None
    current_index_name: Optional[str] = None
    
    def __post_init__(self):
        if self.converstation_history is None:
            self.converstation_history = []


class IntelligentAgent:
    def __init__(self, default_index_name: str = "intelligent-agent-index"):
        self.llm= LLMInitializer.initialize_llm()
        self.state= AgentState()
        self.default_index_name = default_index_name
        
        self.tools=self._create_tools()
        self.agent= self._create_agent()
        
        self.summarization_keywords = [
            "summarize", "summary", "sum up", "brief", "overview", 
            "main points", "key points", "gist", "abstract", "tldr"
        ]
        
        self.conversation_keywords = [
            "tell me about", "explain", "what is", "how does", 
            "discuss", "chat", "talk", "question", "ask"
        ]
        
        self.upload_documents_keywords = [
            "upload", "add documents", "load files", "import", 
            "index documents", "store files", "save documents"
        ]
        
    def _create_tools(self)-> List[Tool]:
        
        def summarize_tool(file_paths: str)-> str:
            try:
                paths=[path.strip() for path in file_paths.split(",")]
                summary= summarize_documents(paths)
                return f"Document Summary:\n{summary}"
            except Exception as e:
                return f"Error summarizing documents: {e}"
            
        def conversation_tool(query: str)-> str:
            try:
                if not self.state.rag_system:
                    return "No documents are currently loaded. Please upload documents first."
                result=self.state.rag_system.query(query)
                return result["änswer"]
            except Exception as e:
                return f"Error in conversation: {str(e)}"
        
        def upload_documents_tool(file_paths:str)-> str:
            try:
                paths = [path.strip() for path in file_paths.split(",")]
                index_name= f"{self.default_index_name}-{len(paths)}-docs"
                self.state.rag_system = create_rag_system(paths, index_name)
                self.state.has_documents = True
                self.state.current_index_name = index_name
                return f"Documents uploaded and indexed under: {index_name}"
            except Exception as e:
                return f"Error uploading documents: {str(e)}"
        
        def detect_intent_tool(user_input: str)-> str:
            intent=self._detect_intent(user_input)
            return f"Detected intent: {intent.value}"
        # Placeholder for tool creation logic
        return [
            Tool(
                name="summarize_documents",
                description="Summarize documents from file paths. Input should be comma-separated file paths.",
                func=summarize_tool
            ),
            Tool(
                name= "conversation_chat",
                description= "Upload documents to vector database for conversation. Input should be comma-separated file paths.",
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
        
    def _detect_intent(self, user_input: str) -> AgentMode:
        user_input=user_input.lower()
        if any(keyword in user_input for keyword in self.summarization_keywords):
            return AgentMode.SUMMARIZATION
        
        if any(keyword in user_input for keyword in self.conversation_keywords):
            return AgentMode.CONVERSATION
        
        if any(keyword in user_input for keyword in self.upload_documents_keywords):
            return AgentMode.DOCUMENT_UPLOAD
        
        return AgentMode.AUTO_DETECT
    
    def _create_agent(self)-> AgentExecutor:
        agent= create_tool_calling_agent(self.llm, self.tools, agent_prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)
    
    def chat(self, user_input: str,file_paths: Optional[List[str]]=None)-> Dict[str, Any]:
        detected_intent = self._detect_intent(user_input)
        if  file_paths:
            file_paths_str = ",".join(file_paths)
            agent_input = f"User message: {user_input}\nFile paths: {file_paths_str}"
        else:
            agent_input = f"User message: {user_input}"
        
        try:
            result= self.agent.invoke({
                "input": agent_input,
            })
            response=result["output"]
            self.state.converstation_history.append({
                "user": user_input,
                "response": response,
                "intent": detected_intent.value,
                "timestamp": self._get_timestamp()
            })
            return {
                "response": response,
                "intent": detected_intent.value,
                "has_documents": self.state.has_documents,
                "conversation_history": self.state.converstation_history[-5:],
                "state": "success",
            }
        except Exception as e:
            error_response = f"I encountered an error: {str(e)}. Please try again or rephrase your request."
            return {
                "response": error_response,
                "intent": detected_intent.value,
                "has_documents": self.state.has_documents,
                "state": "error",
                "error": str(e)
            }
    def direct_summarize(self, file_paths: List[str]) -> str:
        try:
            return summarize_documents(file_paths)
        except Exception as e:
            return f"Error in direct summarization: {str(e)}"
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
            