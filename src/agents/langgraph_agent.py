"""
LangGraph agent implementation with intelligent routing for code assistance.
"""

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Union
from langchain_core.messages import HumanMessage, AIMessage, FunctionMessage, SystemMessage
import json

from ..models.phi_model import PhiModel


class AgentState(TypedDict):
    messages: list[Union[HumanMessage, AIMessage, SystemMessage]]
    chat_state: str


def create_agent(phi_model : PhiModel):
    """Create and compile the LangGraph agent."""
    
    def chat_node(state: AgentState) -> AgentState:
        """Handle normal chat conversations."""
        response = phi_model.generateCodeFromMsg(state["messages"])

        state["messages"].append(AIMessage(content=response))
        print(f"\nAI: '{response}'")

        return state


    def explain_node(state: AgentState) -> AgentState:
        """Handle code explanation requests."""
        state["messages"] += [SystemMessage("""You are a helpful assistant that explains code in simple terms.
                                            However, if no code is provided, you may respond normally""")]

        response = phi_model.generateCodeFromMsg(state["messages"])

        state["messages"].append(AIMessage(content=response))
        print(f"\nAI: '{response}'")

        return state


    def generate_node(state: AgentState) -> AgentState:
        """Handle code generation requests."""
        state["messages"] += [SystemMessage("You are a coding assistant. Generate code that fulfills the user's request.")]

        response = phi_model.generateCodeFromMsg(state["messages"])

        state["messages"].append(AIMessage(content=response))
        print(f"\nAI: {response}")
        return state


    def router(state: AgentState):
        """Route requests to appropriate nodes based on chat state."""
        if state["chat_state"] == "chat_normally":
            return "chat_edge"
        elif state["chat_state"] == "explain_code":
            return "explain_edge"
        elif state["chat_state"] == "generate_code":
            return "generate_edge"
        
    graph = StateGraph(AgentState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("explain_node", explain_node)
    graph.add_node("generate_node", generate_node)
    graph.add_node("router", lambda state: state)

    graph.add_conditional_edges(
        source="router",
        path=router,
        path_map={
            "chat_edge": "chat_node",
            "explain_edge": "explain_node",
            "generate_edge": "generate_node"
        }
    )

    graph.add_edge(START, "router")
    graph.add_edge("chat_node", END)
    graph.add_edge("explain_node", END)
    graph.add_edge("generate_node", END)

    agent = graph.compile()
    
    return agent
