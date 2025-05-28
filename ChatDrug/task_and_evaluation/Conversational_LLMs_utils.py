from openai import OpenAI
from modelscope import snapshot_download
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import openai
import sys
import time
import torch

# client = OpenAI(api_key="sk-c587a4923103469eaf8224f4287ef9d4", base_url="https://api.deepseek.com")
# client = OpenAI(api_key="sk-shbjphcnjideposfsqxsacmgktkkqtzddthrmbcrntbykfdv", base_url="https://api.siliconflow.cn/v1")
client = OpenAI(api_key="sk-1ba964a199d747c89e02cabebf9c7e37", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
openai.api_key = "YOUR_API_KEY"

def complete(messages, conversational_LLM, round_index):
    if conversational_LLM == 'deepseek':
        return complete_deepseek(messages)
    if conversational_LLM == 'chatgpt':
        return complete_chatgpt(messages)
    if conversational_LLM == 'llama':
        return complete_llama(messages)
    if conversational_LLM == 'galactica':
        return complete_galactica(messages, round_index)
    if conversational_LLM == 'dfm':
        return complete_dfm(messages)
    else:
        raise NotImplementedError

def complete_deepseek(messages):
    received = False
    temperature = 0
    print("messages")
    print(messages)
    while not received:
        try:
            response = client.chat.completions.create(
                # model="deepseek-chat",
                # model="deepseek-ai/DeepSeek-V3",
                model="qwen-plus",
                messages=messages,
                temperature=temperature,
                frequency_penalty=0.2,
                n=None
            )
            raw_generated_text = response.choices[0].message.content
            received=True
        except:
            error = sys.exc_info()[0]
            print(f"Error: {error}")
            time.sleep(1)
    return raw_generated_text

def complete_chatgpt(messages):
    received = False
    temperature = 0
    while not received:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", # need modify
                messages=messages,
                temperature=temperature,
                frequency_penalty=0.2,
                n=None)
            raw_generated_text = response["choices"][0]["message"]['content']   
            received=True
        except:
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError: # something is wrong: e.g. prompt too long
                print(f"InvalidRequestError\nPrompt error.\n\n")
                print("prompt too long")
                return "prompt too long"
            if error == AssertionError:
                print("Assert error:", sys.exc_info()[1])
                # assert False
            else:
                print("API error:", error)
            time.sleep(1)
    return raw_generated_text#, messages


global llama_pipe
llama_pipe = None

def init_llama(model_id):
    global llama_pipe
    if llama_pipe is None:
        llama_pipe = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        print("init done")

def complete_llama(messages):
    model_id = snapshot_download("LLM-Research/Meta-Llama-3.1-8B-Instruct", cache_dir='/root/autodl-tmp/Llama-3.1-8B-Instruct-modelscope')
    init_llama(model_id=model_id)
    received = False
    temperature = 0
    while not received:
        # try:
        outputs = llama_pipe(
            messages,
            max_new_tokens=256,
        )
        raw_generated_text = outputs[0]["generated_text"][-1]["content"]
        received = True
        # except:
        #     error = sys.exc_info()[0]
        #     print(f"Error: {error}")
        #     time.sleep(1)
    return raw_generated_text

def messages_2_text(messages):
    # print(messages)
    input_text = ""
    for message in messages:
        content = message["content"]
        input_text += content
    # print("Input text: ", input_text)
    return input_text

global gala_pipe
gala_pipe = None

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

def complete_galactica(messages, round_index):
    init_galactica()
    received = False
    temperature = 0
    if round_index == 0:
        input_text = messages[1]['content']
        input_text = input_text + " [START_I_SMILES]"
    else:
        input_text = ""
        for i in range(len(messages)-1):
            if i%2==0:
                input_text+= messages[i+1]['content']+" [START_I_SMILES]"
            if i%2==1:
                input_text+= messages[i+1]['content']+"[END_I_SMILES]"+"\n\n"
    while not received:
        # try:
        outputs = gala_pipe(
            input_text,
            max_new_tokens=100,
            do_sample=True,
            top_p=0.95,
            temperature=1.0,
            use_cache=True,
            top_k=50,
            repetition_penalty=1.0,
            length_penalty=1,
        )
        output_text = outputs[0]["generated_text"]
        output_text_list = output_text.split("[START_I_SMILES]")
        output_text = output_text_list[2+round_index*3].strip()
        output_text_list = output_text.split("[END_I_SMILES]")
        output_text = output_text_list[0].strip()
        received = True
        # except:
        #     error = sys.exc_info()[0]
        #     print(f"Error: {error}")
        #     time.sleep(1)
    return output_text

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

global dfm_model
global dfm_tokenizer
dfm_model = None
dfm_tokenizer = None

def init_dfm():
    global dfm_model
    global dfm_tokenizer
    
    if (dfm_model is None) or (dfm_tokenizer is None):
        dfm_id = "/root/autodl-tmp/chemdfm"
        dfm_model = AutoModelForCausalLM.from_pretrained(dfm_id, torch_dtype=torch.float16, device_map="auto")
        dfm_tokenizer = AutoTokenizer.from_pretrained(dfm_id)
        print("init done")

def complete_dfm(messages):
    init_dfm()
    global dfm_model
    global dfm_tokenizer
    received = False
    new_messages = transform_messages(messages)
    input_text = formatting_input(current_query=new_messages['current_query'], history=new_messages['history'])
    
    while not received:
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
        received = True

    return generated_text
    

if __name__ == "__main__":
    model_id = "OpenDFM/ChemDFM-v1.5-8B"
    messages_1 = []
    messages_1.append({'role': 'system', 'content': 'You are a helpful chemistry expert with extensive knowledge of drug design. '})
    messages_1.append({'role': 'user', 'content': 'Can you make molecule CCn1c(CC2CC[NH+](Cc3cc(F)cc(F)c3)CC2)n[nH]c1=O more soluble in water? Decrease its logP by at least 0.5. The output molecule should be similar to the input molecule.  Give me five molecules in SMILES only and list them using bullet points. '})
    answer = complete_dfm(messages=messages_1)
    print("*"*100)
    print("Answer 1: ")
    print(answer)
    print("*"*100)
    messages_1.append({'role': 'assistant', 'content': answer})
    messages_1.append({'role': 'user', 'content': "Your provided sequence could not achieve goal. Can you give me new molecules?"})
    answer = complete_dfm(messages=messages_1)
    print("*"*100)
    print("Answer 2: ")
    print(answer)
    print("*"*100)