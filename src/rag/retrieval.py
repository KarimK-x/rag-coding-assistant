"""
Document retrieval functions for RAG system.
"""

from typing import Tuple
from langchain_core.documents import Document


def retrieveDocsFromQuery(vectorstore, query: str, k: int = 4, printResults: bool = False) -> Tuple[list, list]:
    """Takes in a query and returns K most relevant tasks.
    Specifically set up for the vector database set up by ChromaDB for the humaneval dataset."""

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    tasks = []
    solutions = []
    for i, doc in enumerate(retriever.invoke(query)):
        tasks.append(doc.page_content)
        solutions.append(doc.metadata["solution"])

        if printResults:
            print(f"\n==========\nTASK {i+1} IN DATABASE\n==========\n{tasks[i]}", '\n')
            print(f"\n==========\nSOLUTION {i+1} IN DATABASE\n==========\n{solutions[i]}")

    return tasks, solutions
