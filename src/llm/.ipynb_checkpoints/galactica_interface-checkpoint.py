import sys
import time
import asyncio
import nest_asyncio
nest_asyncio.apply()
import transformers
from transformers import AutoTokenizer, OPTForCausalLM
import torch

global gala_pipe
gala_pipe = None

def messages_2_text(messages):
    input_text = ""
    if len(messages) == 2:
        input_text = messages[1]['content']
        input_text = input_text + "[START_I_SMILES]"
    else:
        for i in range(len(messages)-1):
            if i%2 == 0:
                input_text += messages[i+1]['content']+" [START_I_SMILES]"
            if i%2 == 1:
                input_text += messages[i+1]['content']+"[END_I_SMILES]"
    # print(messages)
    # input_text = ""
    # for message in messages:
    #     content = message["content"]
    #     input_text += content
    # print("Input text: ", input_text)
    return input_text

def init_galactica():
    global gala_pipe
    
    if gala_pipe is None:
        # tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-6.7b")
        # model = OPTForCausalLM.from_pretrained("facebook/galactica-6.7b", device_map="auto")
        gala_pipe = transformers.pipeline(
        "text-generation",
        model="facebook/galactica-6.7b",
        device_map="auto",
        torch_dtype="auto"
        )
        print("init done")

async def do_one(messages):
    print("Messages: ", messages)
    messages_text = messages_2_text(messages)
    round_index = int(len(messages)/2 - 1)
    print("Messages text: ", messages_text)
    
    outputs = gala_pipe(
        messages_text,
        max_new_tokens=100,
    )
    output_text = outputs[0]["generated_text"]
    output_text_list = output_text.split("[START_I_SMILES]")
    output_text = output_text_list[2+round_index*2].strip()
    output_text_list = output_text.split("[END_I_SMILES]")
    output_text = output_text_list[0].strip()
    # print("Original Output: ", outputs[0]["generated_text"])
    # print("Clip Output: ", outputs[0]["generated_text"][len(messages_text):])
    # return outputs[0]["generated_text"][len(messages_text):]
    return output_text

async def parallel_gala_text_completion(
    messages
):
    global gala_pipe

    completions = await do_one(messages=messages)
    return completions

async def galactica_text_async_evaluation(
    messages_list
):
    completions = [
        parallel_gala_text_completion(m) for m in messages_list
    ]

    answers = await asyncio.gather(*completions)
    return answers
    

def run_gala_prompts(
    messages_list: list,
    # cache_dir='/root/autodl-tmp/Llama-3.1-8B-Instruct-modelscope'
    # model_id,
    **kwargs
):
    init_galactica()
    # global llama_client
    received = False
    answer_strings = []
    while not received:
        # try:
        async def main():
            print("Galactica Generating...")
            answer_objects = await galactica_text_async_evaluation(messages_list=messages_list)
            print("Galactica Generation Done.")
            for a in answer_objects:
                answer_strings.append(a)
        asyncio.run(main())
        received = True
        # except:
            # error = sys.exc_info()[0]
            # print(f"Error: {error}")
            # time.sleep(1)
            
    return [{"answer": a, "usage": "no usage"} for a in answer_strings]

if __name__ == "__main__":
    messages_list = []
    messages_1 = []
    messages_1.append({'role': 'system', 'content': 'You are a helpful chemistry expert with extensive knowledge of drug design. '})
    # messages_1.append({'role': 'user', 'content': 'Question: Can you make molecule [START_I_SMILES]C1CC[C@@H](CCc2nc([C@@H]3CSCCO3)no2)[NH2+]C1[END_I_SMILES] more soluble in water? The output molecule should be similar to the input molecule.\n'})
    messages_1.append({'role': 'user', 'content': 'Question: Can you make molecule [START_I_SMILES]C1CC[C@@H](CCc2nc([C@@H]3CSCCO3)no2)[NH2+]C1[END_I_SMILES] more soluble in water? The output molecule should be similar to the input molecule.\nAnswer: Yes, Here is an edited SMILES: [START_I_SMILES]'})
    messages_list.append(messages_1)
    answer = run_gala_prompts(messages_list=messages_list)[0]["answer"]
    print("*"*100)
    print("Answer 1: ")
    print(answer)
    print("*"*100)
    messages_2 = messages_1
    messages_2.append({'role': 'assistant', 'content': answer})
    messages_2.append({'role': 'user', 'content': "Question: Your provided sequence could not achieve goal. Can you give me new molecules?Answer: Yes, Here is an different SMILES: [START_I_SMILES]"})
    messages_list.append(messages_2)
    answer = run_gala_prompts(messages_list=messages_list)[1]["answer"]
    print("*"*100)
    print("Answer 2: ")
    print(answer)
    print("*"*100)
    