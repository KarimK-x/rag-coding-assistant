"""
Phi-3.5 model initialization and text generation functions.
"""

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from langchain_core.messages.utils import convert_to_openai_messages


class PhiModel:
    """Wrapper class for Phi-3.5 model operations."""
    
    def __init__(self, model_name="microsoft/Phi-3.5-mini-instruct"):
        """Initialize the Phi-3.5 model and processor."""
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    def generateCodeFromMsg(self, msg, max_new_tokens=500):
        """Generate code/text from a list of messages."""
        msg_converted = convert_to_openai_messages(msg)

        inputs = self.processor.apply_chat_template(
            msg_converted,
            add_generation_prompt=True,   # append the "now you speak" token(s)
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        start = inputs["input_ids"].shape[-1]
        response = self.processor.decode(outputs[0][start:], skip_special_tokens=True)

        return response
