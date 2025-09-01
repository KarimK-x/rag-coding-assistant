"""
Gradio web interface for the RAG coding assistant.
"""

import gradio as gr
from langchain_core.messages import HumanMessage, SystemMessage

from ..agents.langgraph_agent import create_agent
from ..agents.intent_classifier import classify_intent
from ..rag.retrieval import retrieveDocsFromQuery


def gradio_step(user_input, chat_history, conv_history, operation, vectorstore, phi_model):
    """
    Process a single step in the Gradio interface.
    
    Args:
        user_input: str, the new user message
        chat_history: list of (user, bot) tuples for display
        conv_history: List[HumanMessage|SystemMessage|AIMessage], your raw transcript
        operation: str, one of "chat_normally", "explain_code", "generate_code"
        vectorstore: The vector store for RAG retrieval
        
    Returns:
        Updated chat_history, conv_history, operation, and empty user input
    """
    # 1) Handle "new" to reset
    if user_input.lower().strip() == "new":
        return [], [], [], ""

    # 2) Intent classification
    operation = classify_intent(user_input, phi_model)

    # 3) Build up conv_history for RAG
    if operation in ("explain_code", "generate_code"):
        tasks, sols = retrieveDocsFromQuery(vectorstore, user_input, k=3, printResults=False)

        ctx = "\n\n".join(f"Task:\n{t}\n\nSolution:\n{s}" for t, s in zip(tasks, sols))
        conv_history += [
            SystemMessage(f"Use the following examples for context...\n\n{ctx}"),
            HumanMessage(user_input)
        ]
    else:
        conv_history += [
            SystemMessage("You are a helpful assistant"),
            HumanMessage(user_input)
        ]

    # 4) Invoke your LangGraph agent
    agent = create_agent(phi_model)
    result = agent.invoke({
        "messages": conv_history,
        "chat_state": operation
    })
    conv_history = result["messages"]
    operation = result["chat_state"]

    # 5) Extract and append bot reply to chat_history
    bot_msg = conv_history[-1].content
    chat_history = chat_history or []
    chat_history.append((user_input, bot_msg))

    return chat_history, conv_history, operation, ""


def create_gradio_interface(vectorstore, phi_model):
    """
    Create and return the Gradio interface.
    
    Args:
        vectorstore: The vector store for RAG retrieval
        
    Returns:
        Gradio Blocks interface
    """
    with gr.Blocks() as demo:
        gr.Markdown("## PHI3 Coding Assistant")
        chatbot = gr.Chatbot()
        user_in = gr.Textbox(placeholder="Type here and press Enter…")
        conv_history = gr.State([])               # your raw message list
        operation = gr.State("chat_normally")  # initial routing state

        user_in.submit(
            fn=lambda ui, ch, cvh, op: gradio_step(ui, ch, cvh, op, vectorstore, phi_model),
            inputs=[user_in, chatbot, conv_history, operation],
            outputs=[chatbot, conv_history, operation, user_in],
        )
    
    return demo


def launch_gradio_app(vectorstore, phi_model ,share=True, debug=True):
    """
    Launch the Gradio application.
    
    Args:
        vectorstore: The vector store for RAG retrieval
        share: Whether to create a public link
        debug: Whether to enable debug mode
    """
    demo = create_gradio_interface(vectorstore, phi_model)
    demo.launch(share=share, debug=debug)
