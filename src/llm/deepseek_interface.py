import asyncio
import nest_asyncio

nest_asyncio.apply()
import os
import sys
import time

from typing import Union

global deepseek_client
deepseek_client = None

def init_deepseek():
    from openai import AsyncOpenAI
    global deepseek_client
    if deepseek_client is None:
        # deepseek_client = AsyncOpenAI(api_key="sk-c587a4923103469eaf8224f4287ef9d4", base_url="https://api.deepseek.com")
        ### Base Planner
        deepseek_client = AsyncOpenAI(api_key="sk-3z4y46KjqcLSie0hmgVC5JTMNIVUDLPxvYPu8sknbqLPowuH", base_url="https://api.lkeap.cloud.tencent.com/v1",)
        ### LLM Planner
        # deepseek_client = AsyncOpenAI(api_key="sk-zgqsblbwondpoxlactagcfrqidwmlbcikrpldomyqceqpsss", base_url="https://api.siliconflow.cn/v1")
        # deepseek_client = AsyncOpenAI(api_key="sk-shbjphcnjideposfsqxsacmgktkkqtzddthrmbcrntbykfdv", base_url="https://api.siliconflow.cn/v1")
        
        # print("Client Successfully Initialized...")

        
async def parallel_deepseek_text_completion(
#     messages, model="deepseek-chat", **kwargs
# ):
    messages, 
    # model="deepseek-ai/DeepSeek-V3", 
    # model="qwen-plus",
    model="deepseek-v3",
    **kwargs
):
    """Run chat completion calls in parallel"""
    global deepseek_client

    completions = await deepseek_client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs
    )
    return completions

async def deepseek_text_async_evaluation(
#     messages_list, model="deepseek-chat", **kwargs
# ):
    messages_list, 
    # model="deepseek-ai/DeepSeek-V3", 
    model="deepseek-v3",
    **kwargs
):
    completions = [
        parallel_deepseek_text_completion(m) for m in messages_list
    ]
    
    answers = await asyncio.gather(*completions)
    return answers

def run_deepseek_prompts(
    messages_list: list,
    # model="deepseek-chat",
    # model="deepseek-ai/DeepSeek-V3",
    model="deepseek-v3",
    **kwargs
):
    """Run the given prompts with the deepseek interface"""
    init_deepseek()
    received = False
    # Apply defaults to kwargs
    kwargs["temperature"] = kwargs.get("temperature", 0)
    kwargs["frequency_penalty"] = kwargs.get("frequency_penalty", 0.2)
    kwargs["max_tokens"] = kwargs.get("max_tokens", 1024)
    kwargs["stream"] = kwargs.get("stream", False)
    
    answer_strings = []
    usages = []
    print("Messages: ")
    print(messages_list[0])
    while not received:
        try:
            async def main():
                answer_objects = await deepseek_text_async_evaluation(messages_list, model=model, **kwargs)
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


if __name__ == "__main__":
    messages_list = []
    messages_1 = []
    messages_1.append({'role': 'system', 'content': 'You are a helpful chemistry expert with extensive knowledge of drug design. '})
    messages_1.append({'role': 'user', 'content': 'Can you make molecule COC(=O)C(CCCN(C)CCCc1nc2ccccc2[nH]1)(c1ccc(Br)cc1)C(C)C reduce its cardiac toxicity? Decrease its hERG. The output molecule should be similar to the input molecule. Give me five molecules in SMILES only and list them using bullet points. '})
    messages_list = [messages_1]
    answer = run_deepseek_prompts(messages_list=messages_list)[0]["answer"]
    print("*"*100)
    print("Answer 1: ")
    print(answer)
    print("*"*100)
    messages_2 = messages_1
    messages_2.append({'role': 'assistant', 'content': answer})
    messages_2.append({'role': 'user', 'content': "Your provided sequence could not achieve goal. You are suggested to do the modification based on the suggestion: Replace Aromatic Amine with Amide Group. Can you give me new molecules?"})
    messages_list = [messages_2]
    answer = run_deepseek_prompts(messages_list=messages_list)[0]["answer"]
    print("*"*100)
    print("Answer 2: ")
    print(answer)
    print("*"*100)
    messages_3 = messages_2
    messages_3.append({'role': 'assistant', 'content': answer})
    messages_3.append({'role': 'user', 'content': "Your provided sequence could not achieve goal. You are suggested to do the modification based on the suggestion: Replace Ether Linkage with a More Flexible Alkyl Chain. Can you give me new molecules?"})
    messages_list = [messages_3]
    answer = run_deepseek_prompts(messages_list=messages_list)[0]["answer"]
    print("*"*100)
    print("Answer 3: ")
    print(answer)
    print("*"*100)





