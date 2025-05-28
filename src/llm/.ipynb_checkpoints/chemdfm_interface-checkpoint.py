import sys
import time
import asyncio
import nest_asyncio
nest_asyncio.apply()
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch

global dfm_model
global dfm_tokenizer
dfm_model = None
dfm_tokenizer = None

def formatting_input(current_query, history):
    input_text = ''
    for idx, (query, answer) in enumerate(history):
        input_text += f"[Round {idx}]\nHuman: {query}\nAssistant: {answer}\n"
    input_text += f"[Round {len(history)}]\nHuman: {current_query}\nAssistant:"
    return input_text


def transform_messages(messages):
    current_query = ""
    history_query = []
    history_answer = []
    for dialoge in messages:
        if dialoge['role'] == 'system':
            continue
        if dialoge['role'] == 'user':
            history_query.append(dialoge['content'])
        if dialoge['role'] == 'assistant':
            history_answer.append(dialoge['content'])
    assert len(history_query) == (len(history_answer)+1)
    current_query = history_query.pop()
    assert len(history_query) == len(history_answer)
    history_tuple_list = [(history_query[i], history_answer[i]) for i in range(len(history_query))]
    new_messages = {'current_query': current_query, 'history': history_tuple_list}
    return new_messages


def init_dfm():
    global dfm_model
    global dfm_tokenizer
    
    if (dfm_model is None) or (dfm_tokenizer is None):
        dfm_id = "/root/autodl-tmp/chemdfm"
        dfm_model = AutoModelForCausalLM.from_pretrained(dfm_id, torch_dtype=torch.float16, device_map="auto")
        dfm_tokenizer = AutoTokenizer.from_pretrained(dfm_id)
        print("init done")

async def do_one(messages):
    input_text = formatting_input(current_query=messages['current_query'], history=messages['history'])

    inputs = dfm_tokenizer(input_text, return_tensors="pt").to("cuda")
    generation_config = GenerationConfig(
        do_sample=True,
        top_k=20,
        top_p=0.9,
        temperature=0.9,
        max_new_tokens=1024,
        repetition_penalty=1.05,
        eos_token_id=dfm_tokenizer.eos_token_id
    )

    outputs = dfm_model.generate(**inputs, generation_config=generation_config)
    generated_text = dfm_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][len(input_text):]
    
    return generated_text

async def parallel_dfm_text_completion(
    messages
):
    global dfm_model
    global dfm_tokenizer

    new_messages = transform_messages(messages)

    completions = await do_one(messages=new_messages)
    return completions

async def dfm_text_async_evaluation(
    messages_list
):
    completions = [
        parallel_dfm_text_completion(m) for m in messages_list
    ]

    answers = await asyncio.gather(*completions)
    return answers
    

def run_dfm_prompts(
    messages_list: list,
    # cache_dir='/root/autodl-tmp/Llama-3.1-8B-Instruct-modelscope'
    **kwargs
):
    
    init_dfm()
    global dfm_model
    global dfm_tokenizer
    received = False
    answer_strings = []
    while not received:
        # try:
        async def main():
            print("DFM Generating...")
            answer_objects = await dfm_text_async_evaluation(messages_list=messages_list)
            print("DFM Generation Done.")
            for a in answer_objects:
                answer_strings.append(a)
        asyncio.run(main())
        received = True
        # except:
        #     error = sys.exc_info()[0]
        #     print(f"Error: {error}")
        #     time.sleep(1)
    
    return [{"answer": a, "usage": "no usage"} for a in answer_strings]

if __name__ == "__main__":
    messages_list = []
    messages_1 = []
    messages_1.append({'role': 'system', 'content': 'You are a helpful chemistry expert with extensive knowledge of drug design. '})
    messages_1.append({'role': 'user', 'content': 'Can you make molecule Cc1ccc([C@@H](NC(=O)NCCO)c2ccccc2)cc1 more like a drug and increase QED by at least 0.1. ? The output molecule should be similar to the input molecule.  Give me five molecules in SMILES only and list them using bullet points. No explanation is needed. '})
    messages_list.append(messages_1)
    answer = run_dfm_prompts(messages_list=messages_list)[0]["answer"]
    print("*"*100)
    print("Answer 1: ")
    print(answer)
    print("*"*100)
    messages_2 = messages_1
    messages_2.append({'role': 'assistant', 'content': answer})
    messages_2.append({'role': 'user', 'content': "Your provided sequence could not achieve goal. Can you give me new molecules?"})
    messages_list.append(messages_2)
    answer = run_dfm_prompts(messages_list=messages_list)[1]["answer"]
    print("*"*100)
    print("Answer 2: ")
    print(answer)
    print("*"*100)
    