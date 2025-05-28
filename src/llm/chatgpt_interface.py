import asyncio
import nest_asyncio

nest_asyncio.apply()
import os
import sys
import time

from typing import Union

import openai

global openai_client
openai_client = None

def init_openai():
    global openai_client
    if openai_client is None:
        openai_client = openai.AsyncOpenAI()
        openai.api_key = os.getenv("OPENAI_API_KEY_DEV")
        print("Client Successfully Initialized...")
        
async def parallel_openai_text_completion(
    messages, model="text-davinci-003", **kwargs
):
    """Run chat completion calls in parallel"""
    global openai_client

    completions = await openai_client.chat.completions.create(
        messages=messages, model=model, **kwargs
    )

    return completions

async def openai_text_async_evaluation(
    messages_list, model="text-davinci-003", **kwargs
):
    completions = [
        parallel_openai_text_completion(m) for m in messages_list
    ]
    
    answers = await asyncio.gather(*completions)
    return answers

def run_openai_prompts(
    messages_list: list,
    model="gpt-4",
    **kwargs
):
    """Run the given prompts with the deepseek interface"""
    init_openai()
    received = False
    # Apply defaults to kwargs
    kwargs["temperature"] = kwargs.get("temperature", 0)
    kwargs["frequency_penalty"] = kwargs.get("frequency_penalty", 0.2)
    kwargs["max_tokens"] = kwargs.get("max_tokens", 1024)
    kwargs["stream"] = kwargs.get("stream", False)
    
    answer_strings = []
    usages = []
    while not received:
        try:
            async def main():
                print("Messages sample: ")
                print(messages_list[0])
                answer_objects = await openai_text_async_evaluation(messages_list, model=model, **kwargs)
                for a in answer_objects:
                    answer_strings.append(a.choices[0].message.content)
                    usages.append({"completion_tokens": a.usage.completion_tokens, "prompt_tokens": a.usage.prompt_tokens})
            
            asyncio.run(main())
            received=True
        except:
            error = sys.exc_info()[0]
            print(f"Error: {error}")
            time.sleep(1)
        
    return [{"answer": a, "usage": u} for a, u in zip(answer_strings, usages)]

    
_test_prompt = (
    "What are the top-3 catalysts that perform the hydrodeoxygenation reaction and demonstrate higher adsorption energy for acetate?. You should include candidate catalysts with the following properties: high conversion. Provide scientific explanations for each of the catalysts. Finally, return a python list named final_answer which contains the top-5 catalysts."
    "Take a deep breath and let's think step-by-step. Remember, you need to return a python list named final_answer!"
)
_test_system_prompt = (
    "You are a helpful chemistry expert with extensive knowledge of "
    "catalysis. You will give recommendations for catalysts, including "
    "chemically accurate descriptions of the interaction between the catalysts "
    "and adsorbate(s). Make specific recommendations for catalysts, including "
    "their chemical composition. Make sure to follow the formatting "
    "instructions. Do not provide disclaimers or notes about your knowledge of "
    "catalysis."
)
if __name__ == "__main__":
    print(run_openai_prompts([_test_prompt] * 20, [_test_system_prompt] * 20))