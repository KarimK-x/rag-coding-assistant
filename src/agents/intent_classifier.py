"""
Intent classification system for smart routing of user requests.
"""

import json
from langchain_core.messages import SystemMessage
from langchain_core.messages.utils import convert_to_openai_messages

from ..models.phi_model import PhiModel


def classify_intent(user_input: str, llm : PhiModel) -> str:
    """
    Classify user intent for smart routing.
    
    Args:
        user_input: The user's message
        
    Returns:
        One of: "explain_code", "generate_code", "chat_normally"
    """
    intent_classifier_prompt = f"""
        You are an intent‑classifier. Read the user's message and respond *only* with JSON in this exact schema:
        {{
        "task": <one of: "explain_code", "generate_code", "chat_normally">,
        "user_input": <the original user text, verbatim>
        }}

        Examples:
        User: "Can you walk me through what this function does line by line?"
        ➞ {{"task":"explain_code","user_input":"Can you walk me through what this function does line by line?"}}

        User: "Write me a Python script that parses a CSV and prints the average of column A."
        ➞ {{"task":"generate_code","user_input":"Write me a Python script that parses a CSV and prints the average of column A."}}

        User: "Hey, how's your day going?"
        ➞ {{"task":"chat_normally","user_input":"Hey, how's your day going?"}}

        Now classify:
        User: "{user_input}"
        """
    
    input_to_intent_classifier = llm.processor.apply_chat_template(
        convert_to_openai_messages([SystemMessage(intent_classifier_prompt)]),
        add_generation_prompt=True,   # append the "now you speak" token(s)
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(llm.model.device)

    output = llm.model.generate(**input_to_intent_classifier, max_new_tokens=200)
    start = input_to_intent_classifier["input_ids"].shape[-1]
    response = llm.processor.decode(output[0][start:], skip_special_tokens=True)
    
    print(f"\nRESPONSE OF INTENT CLASIFIER IS {response}\n")

    try:
        parsed_response = json.loads(response)
        operation = parsed_response["task"]
        return operation
    except json.JSONDecodeError:
        print(f"[Warning] Failed to parse intent JSON: {repr(response)}")
        return "chat_normally"

