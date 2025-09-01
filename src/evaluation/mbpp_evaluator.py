"""
MBPP evaluation system for testing code generation capabilities.
"""

import pandas as pd
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.messages.utils import convert_to_openai_messages


def load_mbpp_data(file_path: str, sample_size: int = 10, random_state: int = 42):
    """
    Load and sample MBPP dataset.
    
    Args:
        file_path: Path to the MBPP JSON file
        sample_size: Number of samples to take
        random_state: Random seed for reproducibility
        
    Returns:
        Sampled MBPP DataFrame
    """
    mbpp_df = pd.read_json(file_path)
    mbpp_df = mbpp_df.sample(sample_size, random_state=random_state).loc[:, ["prompt", "task_id", "code", "test_list"]]
    mbpp_df.set_index("task_id", inplace=True, drop=True)
    return mbpp_df


def evaluateRAG(query: str, vectorstore):
    """
    Evaluate RAG system performance on a given query.
    
    Args:
        query: The programming task query
        vectorstore: The vector store for retrieval
        
    Returns:
        Generated code response
    """
    retriever = vectorstore.as_retriever()
    docs = retriever.invoke(query)
    tasks = []
    solutions = []

    for doc in docs:
        tasks.append(doc.page_content)
        solutions.append(doc.metadata["solution"])

    context = "\n\n".join(
        f"Task:\n{task}\n\nSolution:\n{solution}"
        for task, solution in zip(tasks, solutions)
    )

    context_msg = (
        "Use the following examples for context:\n\n"
        f"{context}"
    )

    instruction_msg = (
        f"""You are an expert python programmer. Here is your task: \n{query}\n Respond *only* with the python function.No more. Do NOT add any additional text after the function's end:
        """
    )

    msg = []
    msg.append(SystemMessage(context_msg))
    msg.append(HumanMessage(instruction_msg))

    input = processor.apply_chat_template(
        convert_to_openai_messages(msg),
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    output = model.generate(**input, max_new_tokens=500)
    start = input["input_ids"].shape[-1]
    response = processor.decode(output[0][start:], skip_special_tokens=True)

    return response


def generate_model_outputs(mbpp_df, vectorstore):
    """
    Generate model outputs for all prompts in the MBPP dataset.
    
    Args:
        mbpp_df: MBPP DataFrame
        vectorstore: The vector store for retrieval
        
    Returns:
        Pandas Series with model outputs
    """
    model_outputs = []
    for prompt in mbpp_df["prompt"]:
        output = evaluateRAG(prompt, vectorstore)
        model_outputs.append(output)

    return pd.Series(model_outputs)


def test_code_execution(code: str, test_cases: list):
    """
    Test if generated code passes the given test cases.
    
    Args:
        code: The generated Python code
        test_cases: List of test case strings
        
    Returns:
        Boolean indicating if all tests passed
    """
    try:
        # Execute the code
        exec(code)
        
        # Run test cases
        for test in test_cases:
            exec(test)
        
        print("No Assertion Error. All tests Passed")
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        return False


def run_evaluation_example(mbpp_df, vectorstore, index: int = 0):
    """
    Run a single evaluation example.
    
    Args:
        mbpp_df: MBPP DataFrame
        vectorstore: The vector store for retrieval
        index: Index of the example to run
    """
    prompt = mbpp_df['prompt'].iloc[index]
    expected_code = mbpp_df['code'].iloc[index]
    test_cases = mbpp_df["test_list"].iloc[index]
    
    print(f"\nThe Task is \n{prompt}\n. The given solution in docs is \n{expected_code}\n.")
    
    generated_output = evaluateRAG(prompt, vectorstore)
    print(f"The output generated is: \n{generated_output}")
    
    # Test the generated code
    print(f"\nTesting generated code with test cases:")
    for test in test_cases:
        print(test)
    
    test_result = test_code_execution(generated_output, test_cases)
    return test_result
