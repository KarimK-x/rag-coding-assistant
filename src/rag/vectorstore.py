"""
Vector store setup and management using ChromaDB and HuggingFace embeddings.
"""

import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

#TODO: Standardize setting up the vectorstore for any dataframe.
def setup_vectorstore(df, persist_directory="./data/chroma_store"):
    """
    Set up VectorDB using Chroma with HuggingFace embeddings.
    
    Args:
        df: DataFrame with columns 'prompt', 'canonical_solution', 'task_id'
        persist_directory: Directory to persist the vector store
    
    Returns:
        Chroma vectorstore instance
    """
    # Setting up embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

    # Setting up data for compatibility with Chroma.from_texts
    prompts = [prompt for prompt in df["prompt"]]
    metadata = [{"solution": sol} for sol in df["canonical_solution"]]

    # Creating Vectorstore
    vectorstore = Chroma.from_texts(
        texts=prompts,
        embedding=embeddings,
        metadatas=metadata,
        ids=df["task_id"],
        persist_directory=persist_directory,
    )
    
    return vectorstore


def setup_mbpp_vectorstore(mbpp_df, persist_directory="./data/chroma_store2"):
    """
    Set up VectorDB for MBPP dataset evaluation.
    
    Args:
        mbpp_df: MBPP DataFrame with columns 'prompt', 'code'
        persist_directory: Directory to persist the vector store
    
    Returns:
        Chroma vectorstore instance
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    
    prompts_mbpp = [prompt for prompt in mbpp_df["prompt"]]
    metadatas_mbpp = [{"solution": code} for code in mbpp_df["code"]]

    vectorstore2 = Chroma.from_texts(
        texts=prompts_mbpp,
        embedding=embeddings,
        metadatas=metadatas_mbpp,
        persist_directory=persist_directory
    )
    
    return vectorstore2
