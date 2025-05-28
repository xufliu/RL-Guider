import sys
import time
import asyncio
import nest_asyncio
nest_asyncio.apply()
import transformers
import torch

global llama_pipe
llama_pipe = None

def init_llama(model_id):
    global llama_pipe
    
    if llama_pipe is None:
        from modelscope import snapshot_download
        model_id = snapshot_download("LLM-Research/Meta-Llama-3.1-8B-Instruct", cache_dir='/root/autodl-tmp/Llama-3.1-8B-Instruct-modelscope')
        llama_pipe = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        print("init done")

async def do_one(messages):
    outputs = llama_pipe(
        messages,
        max_new_tokens=1024,
    )
    return outputs[0]["generated_text"][-1]

async def parallel_llama_text_completion(
    messages
):
    global llama_pipe

    completions = await do_one(messages=messages)
    return completions

async def llama_text_async_evaluation(
    messages_list
):
    completions = [
        parallel_llama_text_completion(m) for m in messages_list
    ]

    answers = await asyncio.gather(*completions)
    return answers
    

def run_llama_prompts(
    messages_list: list,
    # cache_dir='/root/autodl-tmp/Llama-3.1-8B-Instruct-modelscope'
    model_id,
    **kwargs
):
    
    init_llama(model_id=model_id)
    global llama_client
    received = False
    answer_strings = []
    while not received:
        try:
            async def main():
                print("LLaMA Generating...")
                answer_objects = await llama_text_async_evaluation(messages_list=messages_list)
                print("LLaMA Generation Done.")
                for a in answer_objects:
                    answer_strings.append(a)
            asyncio.run(main())
            received = True
        except:
            error = sys.exc_info()[0]
            print(f"Error: {error}")
            time.sleep(1)

        
    
    return [{"answer": a, "usage": "no usage"} for a in answer_strings]

if __name__ == "__main__":
    model_id = snapshot_download("LLM-Research/Meta-Llama-3.1-8B-Instruct", cache_dir='/root/autodl-tmp/Llama-3.1-8B-Instruct-modelscope')
    messages_list = []
    messages_1 = []
    messages_1.append({'role': 'system', 'content': 'You are a helpful chemistry expert with extensive knowledge of drug design. '})
    messages_1.append({'role': 'user', 'content': 'Can you make molecule CCn1c(CC2CC[NH+](Cc3cc(F)cc(F)c3)CC2)n[nH]c1=O more soluble in water? Decrease its logP by at least 0.5. The output molecule should be similar to the input molecule.  Give me five molecules in SMILES only and list them using bullet points. '})
    messages_list.append(messages_1)
    answer = run_llama_prompts(messages_list=messages_list, model_id=model_id)[0]["answer"]['content']
    print("*"*100)
    print("Answer 1: ")
    print(answer)
    print("*"*100)
    messages_2 = messages_1
    messages_2.append({'role': 'assistant', 'content': answer})
    messages_2.append({'role': 'user', 'content': "Your provided sequence could not achieve goal. Can you give me new molecules?"})
    messages_list.append(messages_2)
    answer = run_llama_prompts(messages_list=messages_list, model_id=model_id)[1]["answer"]['content']
    print("*"*100)
    print("Answer 2: ")
    print(answer)
    print("*"*100)
    