"""
LangGraph State Machine for AutoStream Conversational AI Agent
Orchestrates intent classification, RAG responses, and lead capture
"""

import os
from typing import TypedDict, List, Optional
from dotenv import load_dotenv
import google.generativeai as genai
from langgraph.graph import StateGraph, END
from langgraph.graph.graph import Graph

from intent_classifier import get_intent_classifier, Intent
from rag import get_rag_pipeline

# Load environment variables
load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

genai.configure(api_key=GOOGLE_API_KEY)


# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """
    State object that persists across conversation turns
    """
    messages: List[dict]              # Conversation history
    intent: Optional[str]             # Latest classified intent
    response: Optional[str]           # Response to send to user
    collected_info: dict              # {name, email, platform}
    current_field_collecting: Optional[str]  # Which field we're asking for
    lead_captured: bool               # Has lead been captured?
    turn_count: int                   # Track conversation turns


# ============================================================================
# NODES
# ============================================================================

def intent_classifier_node(state: AgentState) -> dict:
    """
    Node 1: Classify the user's intent
    
    Extracts the latest user message and classifies it.
    """
    # Get the latest user message
    latest_message = state["messages"][-1]
    user_text = latest_message["content"]
    
    # Classify intent
    classifier = get_intent_classifier()
    intent = classifier.classify(user_text)
    
    print(f"  [Intent Classifier] Message: \"{user_text[:50]}...\"")
    print(f"  [Intent Classifier] → Intent: {intent.value}")
    
    return {
        "intent": intent.value,
        "response": None
    }


def casual_responder_node(state: AgentState) -> dict:
    """
    Node 2: Handle casual greetings
    
    Responds warmly to casual messages and asks how to help.
    """
    casual_responses = [
        "👋 Hey there! How can I help you with AutoStream today?",
        "😊 Hello! What would you like to know about our video editing platform?",
        "Welcome! Feel free to ask about our pricing, features, or anything else!",
        "Hi! Happy to help. What can I tell you about AutoStream?",
    ]
    
    import random
    response = random.choice(casual_responses)
    
    print(f"  [Casual Responder] → {response}")
    
    return {"response": response}


def rag_responder_node(state: AgentState) -> dict:
    """
    Node 3: Respond to product/pricing inquiries using RAG
    
    Retrieves from knowledge base and generates grounded answer.
    """
    latest_message = state["messages"][-1]
    user_query = latest_message["content"]
    
    # Use RAG pipeline
    rag = get_rag_pipeline()
    result = rag.answer_question(user_query, verbose=False)
    response = result["answer"]
    
    print(f"  [RAG Responder] Query: \"{user_query[:50]}...\"")
    print(f"  [RAG Responder] → {response[:80]}...")
    
    return {"response": response}


def field_collector_node(state: AgentState) -> dict:
    """
    Node 4: Collect user information for lead capture
    
    Extracts name, email, and platform one at a time across turns.
    """
    latest_message = state["messages"][-1]
    user_text = latest_message["content"]
    
    collected = state["collected_info"].copy()
    current_field = state["current_field_collecting"]
    
    # If no field is being collected yet, start with name
    if current_field is None:
        response = "Great! I'd love to help you get started. 🚀\n\nWhat's your name?"
        next_field = "name"
    
    # Collect name
    elif current_field == "name":
        collected["name"] = user_text.strip()
        response = f"Nice to meet you, {collected['name']}! 👋\n\nWhat's your email address?"
        next_field = "email"
        print(f"  [Field Collector] Collected name: {collected['name']}")
    
    # Collect email
    elif current_field == "email":
        collected["email"] = user_text.strip()
        response = f"Thanks! I've got your email as {collected['email']}. 📧\n\nWhich platform are you creating content for? (e.g., YouTube, Instagram, TikTok)"
        next_field = "platform"
        print(f"  [Field Collector] Collected email: {collected['email']}")
    
    # Collect platform
    elif current_field == "platform":
        collected["platform"] = user_text.strip()
        response = "Perfect! Let me capture this information."
        next_field = None  # All fields collected
        print(f"  [Field Collector] Collected platform: {collected['platform']}")
    
    else:
        response = "I'm not sure what I should be collecting. Let me start over."
        next_field = "name"
    
    return {
        "response": response,
        "collected_info": collected,
        "current_field_collecting": next_field
    }


def tool_executor_node(state: AgentState) -> dict:
    """
    Node 5: Execute tool - Call mock_lead_capture when all info collected
    
    Guards against premature execution - only runs when all 3 fields present.
    """
    collected = state["collected_info"]
    
    # Guard: Check all fields are present
    if not all([collected.get("name"), collected.get("email"), collected.get("platform")]):
        response = "❌ Error: Not all information collected. Cannot capture lead."
        print(f"  [Tool Executor] Guard failed - missing fields")
        return {"response": response, "lead_captured": False}
    
    # All fields present - execute tool
    print(f"  [Tool Executor] Executing: mock_lead_capture(...)")
    print(f"    - Name: {collected['name']}")
    print(f"    - Email: {collected['email']}")
    print(f"    - Platform: {collected['platform']}")
    
    result = mock_lead_capture(
        name=collected["name"],
        email=collected["email"],
        platform=collected["platform"]
    )
    
    response = result
    
    return {
        "response": response,
        "lead_captured": True,
        "current_field_collecting": None
    }


# ============================================================================
# CONDITIONAL ROUTING EDGES
# ============================================================================

def route_after_classification(state: AgentState) -> str:
    """
    Route based on classified intent
    """
    intent = state["intent"]
    
    if intent == "casual":
        return "casual_responder"
    elif intent == "inquiry":
        return "rag_responder"
    elif intent == "high_intent":
        return "field_collector"
    else:
        return "rag_responder"  # Default fallback


def route_after_field_collection(state: AgentState) -> str:
    """
    Route based on whether all fields are collected
    """
    collected = state["collected_info"]
    current_field = state["current_field_collecting"]
    
    # If we just collected the platform field, we have all 3
    if (collected.get("name") and 
        collected.get("email") and 
        collected.get("platform") and
        current_field is None):
        return "tool_executor"
    else:
        # Still collecting fields
        return END


# ============================================================================
# TOOL: Mock Lead Capture
# ============================================================================

def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """
    Mock function that simulates lead capture
    
    In production, this would save to CRM/database
    """
    print(f"\n✅ LEAD CAPTURE SUCCESS!")
    print(f"   Name: {name}")
    print(f"   Email: {email}")
    print(f"   Platform: {platform}\n")
    
    return f"🎉 Excellent! I've captured your information:\n- Name: {name}\n- Email: {email}\n- Platform: {platform}\n\nYou should receive a welcome email shortly. Let's get you started with AutoStream! 🚀"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def build_agent_graph():
    """
    Build and compile the LangGraph state machine
    """
    # Create graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("intent_classifier", intent_classifier_node)
    graph.add_node("casual_responder", casual_responder_node)
    graph.add_node("rag_responder", rag_responder_node)
    graph.add_node("field_collector", field_collector_node)
    graph.add_node("tool_executor", tool_executor_node)
    
    # Add edges
    
    # Entry point: Start with intent classification
    graph.set_entry_point("intent_classifier")
    
    # After classification, route based on intent
    graph.add_conditional_edges(
        "intent_classifier",
        route_after_classification,
        {
            "casual_responder": "casual_responder",
            "rag_responder": "rag_responder",
            "field_collector": "field_collector",
        }
    )
    
    # Casual responder ends the turn
    graph.add_edge("casual_responder", END)
    
    # RAG responder ends the turn
    graph.add_edge("rag_responder", END)
    
    # Field collector routes to tool executor if complete, otherwise ends turn
    graph.add_conditional_edges(
        "field_collector",
        route_after_field_collection,
        {
            "tool_executor": "tool_executor",
            END: END
        }
    )
    
    # Tool executor ends the turn
    graph.add_edge("tool_executor", END)
    
    # Compile the graph
    compiled_graph = graph.compile()
    
    return compiled_graph


# ============================================================================
# AGENT INTERFACE
# ============================================================================

class AutoStreamAgent:
    """
    Main agent interface for conversation
    """
    
    def __init__(self):
        """Initialize the agent"""
        self.graph = build_agent_graph()
        self.state = {
            "messages": [],
            "intent": None,
            "response": None,
            "collected_info": {},
            "current_field_collecting": None,
            "lead_captured": False,
            "turn_count": 0
        }
    
    def chat(self, user_message: str) -> str:
        """
        Process a user message and return agent response
        
        Args:
            user_message: User's input
            
        Returns:
            Agent's response
        """
        self.state["turn_count"] += 1
        
        # Add user message to history
        self.state["messages"].append({
            "role": "user",
            "content": user_message
        })
        
        print(f"\n{'='*70}")
        print(f"TURN {self.state['turn_count']}")
        print(f"{'='*70}")
        print(f"👤 User: {user_message}")
        print()
        
        # Run the graph
        result_state = self.graph.invoke(self.state)
        
        # Update internal state
        self.state = result_state
        
        # Extract response
        response = result_state["response"]
        
        # Add agent response to history
        self.state["messages"].append({
            "role": "assistant",
            "content": response
        })
        
        print(f"🤖 Agent: {response}")
        print(f"\n[State] Intent: {self.state['intent']}")
        print(f"[State] Collected: {self.state['collected_info']}")
        print(f"[State] Lead Captured: {self.state['lead_captured']}")
        
        return response
    
    def get_state(self) -> dict:
        """Get current agent state"""
        return self.state.copy()


# ============================================================================
# DEMO CONVERSATION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("AUTOSTREAM CONVERSATIONAL AI AGENT - DEMO")
    print("="*70)
    
    agent = AutoStreamAgent()
    
    # Simulated conversation
    conversation = [
        "Hey, how's it going?",  # CASUAL
        "What's the price of the Pro plan?",  # INQUIRY
        "I want to sign up for YouTube content editing!",  # HIGH_INTENT
        "My name is Arthur",  # NAME collection
        "arthur@email.com",  # EMAIL collection
        "YouTube",  # PLATFORM collection
    ]
    
    for user_input in conversation:
        agent.chat(user_input)
        print()
    
    print("\n" + "="*70)
    print("CONVERSATION COMPLETE")
    print("="*70)
    print(f"\nFinal State:")
    print(f"  Turns: {agent.state['turn_count']}")
    print(f"  Lead Captured: {agent.state['lead_captured']}")
    print(f"  Collected Info: {agent.state['collected_info']}")