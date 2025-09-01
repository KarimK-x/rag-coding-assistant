"""
Main application entry point for the RAG Coding Assistant.
"""

import argparse
import pandas as pd
from pathlib import Path

from .models.phi_model import PhiModel
from .rag.vectorstore import setup_vectorstore, setup_mbpp_vectorstore
from .rag.retrieval import retrieveDocsFromQuery
from .agents.langgraph_agent import create_agent
from .agents.intent_classifier import classify_intent
from .ui.gradio_interface import launch_gradio_app
from .evaluation.mbpp_evaluator import load_mbpp_data, run_evaluation_example
from langchain_core.messages import HumanMessage, SystemMessage


def setup_data():
    """Load and prepare the datasets."""
    print("Loading OpenAI HumanEval dataset...")
    df = pd.read_parquet("hf://datasets/openai/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet")
    df = df.iloc[:, :3]
    return df


def run_cli_mode(vectorstore, phi_model):
    """Run the command-line interface mode."""
    conversation_history = []
    operation = "chat"
    agent = create_agent(phi_model)
    
    print("""WELCOME TO PHI 3 CODING ASSISTANT:
          Type 'new' at any time to start a new Chat.
          Type 'exit' to exit program.""")

    while True:
        user_input = input("Enter: ")
        if user_input == "exit":
            break

        print(f"\n\nUser: {user_input}")
        if user_input.lower() == "new":
            conversation_history = []
            docs = ''
            continue

        # Smart routing implementation
        operation = classify_intent(user_input, phi_model)

        if operation == "explain_code" or operation == "generate_code":
            tasks, solutions = retrieveDocsFromQuery(vectorstore, user_input, k=3, printResults=False)

            context = "\n\n".join(
                f"Task:\n{task}\n\nSolution:\n{solution}"
                for task, solution in zip(tasks, solutions)
            )
            system_msg = (
                "Use the following examples for context (do not just repeat them word‑for‑word):\n\n"
                f"{context}"
            )
            conversation_history.append(SystemMessage(system_msg))
            conversation_history.append(HumanMessage(user_input))
        elif operation == "chat_normally":
            conversation_history.append(SystemMessage("You are a helpful assistant"))
            conversation_history.append(HumanMessage(user_input))
        else:
            print("\nINCORRECT OPERATION\n")
            continue

        print(f"Conversation History: {conversation_history}")
        result = agent.invoke({"messages": conversation_history,
                             "chat_state": operation})

        conversation_history = result["messages"]
        operation = result["chat_state"]


def run_gradio_mode(vectorstore, phi_model):
    """Run the Gradio web interface mode."""
    print("Launching Gradio interface...")
    launch_gradio_app(vectorstore,phi_model ,share=True, debug=True)


#TODO: Add run_evaluation_mode

def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="RAG Coding Assistant")
    parser.add_argument(
        "--mode", 
        choices=["cli", "gradio", "evaluation"], 
        default="cli",
        help="Mode to run the application in"
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Directory to store vector databases"
    )
    
    args = parser.parse_args()
    
    # Ensure data directory exists
    Path(args.data_dir).mkdir(exist_ok=True)
    
    # Initialize the model
    print("Initializing Phi-3.5 model...")
    phi_model = PhiModel()
    
    # Setup vector store
    print("Setting up vector store...")
    df = setup_data()
    vectorstore = setup_vectorstore(df, persist_directory=f"{args.data_dir}/chroma_store")
    
    # Run appropriate mode
    if args.mode == "cli":
        run_cli_mode(vectorstore, phi_model)
    elif args.mode == "gradio":
        run_gradio_mode(vectorstore, phi_model)


if __name__ == "__main__":
    main()
