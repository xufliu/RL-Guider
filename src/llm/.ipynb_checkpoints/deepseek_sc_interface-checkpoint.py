import asyncio
import logging
import os

from typing import Union

from openai import AsyncOpenAI

logging.getLogger().setLevel(logging.INFO)

global deepseek_client
deepseek_client = None

def init_deepseek():
    global deepseek_client
    if deepseek_client is None:
        deepseek_client = AsyncOpenAI(api_key="sk-c587a4923103469eaf8224f4287ef9d4", base_url="https://api.deepseek.com")
        print("Client Successfully Initialized...")

        
async def parallel_deepseek_text_completion(
    prompt, model="deepseek-chat", **kwargs
):
    """Run chat completion calls in parallel"""
    global deepseek_client
    
    messages = prompt
    
    completions = await deepseek_client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs
    )
    return completions

async def deepseek_text_async_evaluation(
    prompts, model="deepseek-chat", **kwargs
):
    
    completions = [
        parallel_deepseek_text_completion(p) for i, p in enumerate(prompts)
    ]
    
    answers = await asyncio.gather(*completions)
    return answers

def run_deepseek_sc_prompts(
    prompts: list[list[str]],
    model="deepseek-chat",
    **kwargs
):
    """Run the given prompts with the deepseek interface"""
    init_deepseek()
    # Apply defaults to kwargs
    kwargs["temperature"] = kwargs.get("temperature", 0.6)
    # kwargs["top_p"] = kwargs.get("top_p", 0.3)
    kwargs["max_tokens"] = kwargs.get("max_tokens", 1024)
    kwargs["stream"] = kwargs.get("stream", False)
    
    answer_strings = []
    usages = []
    
    async def main():
        answer_objects = await deepseek_text_async_evaluation(prompts, model=model, **kwargs)
        for a in answer_objects:
            answer_strings.append(a.choices[0].message)
            usages.append({"completion_tokens": a.usage.completion_tokens, "prompt_tokens": a.usage.prompt_tokens})
    
    asyncio.run(main())
    
    return [{"answer": a, "usage": u} for a, u in zip(answer_strings, usages)]

class deepseekInterface:
    """A class to handle communicating with deepseek"""
    
    def __init__(self, model='deepseek-chat'):
        self.model = model
    
    def __call__(
        self,
        prompts: list[list[str]],
        system_prompts: list[Union[str, None]] = None,
        **kwargs,
    ):
        """Run the given prompts with deepseek interface"""
        client = init_deepseek()
        
        kwargs["temperature"] = kwargs.get("temperature", 0.6)
        kwargs["max_tokens"] = kwargs.get("max_tokens", 1024)
        kwargs["stream"] = kwargs.get("stream", False)
        
        if system_prompts is not None:
            system_prompts = [None] * len(prompts)
            
        answer_strings = []
        usages = []

        async def main():
            answer_objects = await deepseek_text_async_evaluation(prompts, system_prompts=system_prompts, model=model, **kwargs)
            for a in answer_objects:
                answer_strings.append(a.choices[0].message)
                usages.append({"completion_tokens": a.usage.completion_tokens, "prompt_tokens": a.usage.prompt_tokens})

        asyncio.run(main())

        return [{"answer": a, "usage": u} for a, u in zip(answer_strings, usages)]

    
_test_prompt = [
    ("Can you make molecule C=C(C)Cc1ccccc1Br more soluble in the water?The output molecule should be similar to the input molecule.  Your provided sequence C=C(COOH)Cc1ccccc1OH, C=C(C)Cc1c(OH)cccc1Br is not correct. Can you give me a new molecule? Let's think step by step. Return a python dict whose key is final_answer and value contains the top-1 drug molecule. Remember, you need to return a python dict!"),
    # ("Your generated SMILES string in final_answer is invalid, please provide a valid one which is similar to the molecule being modified.")
]
_test_system_prompt = (
    "You are a helpful chemistry expert with extensive knowledge of drug design. "
    "You will give recommendations for drug editing, where the drug is represented by valid SMILES string."
)
if __name__ == "__main__":
    print(run_deepseek_prompts([_test_prompt] * 1, [_test_system_prompt] * 1))