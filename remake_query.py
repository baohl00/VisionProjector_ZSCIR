import json
import time
import traceback

from tqdm import tqdm
import openai
from pathlib import Path

# pip install accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig 
import torch

model_path = "/home/support/llm/gemma-2-27b-it"
#model_path = "unsloth/gemma-2-9b-it-bnb-4bit"
#config = AutoConfig.from_pretrained(model_path + "/config.json")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
            model_path, 
 #           config = config)
            device_map="auto",
            torch_dtype=torch.bfloat16)

def generated_query(prompt = None):
        #img_content = 'a woman with knife',
        #instruction = 'a man instead of woman, and wearing red shirt'):
    #input_text = prompt.format(content = img_content, instruction = instruction)
    input_text = prompt 
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

    outputs = model.generate(**input_ids, max_new_tokens=128)
    return preprocess_prompt(tokenizer.decode(outputs[0]))

def preprocess_prompt(prompt = None):
    prompt = prompt.split("\n")
    prompt = [line for line in prompt if "Edited Description:" in line]
    prompt = prompt[-1]
    prompt = prompt.split(": ")[1]
    prompt = prompt.split(".")[0]
    return prompt

#openai.api_key = "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"
#openai.api_key = "sk-svcacct-uNGxMY0V_GQ5C_HwV77vl16CRI4VM4rZOrlqnteP0hAhOixCJGPKI3T3BlbkFJYF84WbYm-a18TYPvqxzorCnIbQpbpHORLzcXuNoR67ex6eswas-TYA"

DATASET = 'circo' # 'cirr', 'fashioniq'

if DATASET == 'circo':
    SPLIT = 'test'
    input_json = './data/circo/annotations/test2.json'
    dataset_path = Path('CIRCO')
elif DATASET == 'cirr':
    SPLIT = 'test1'
    input_json = './data/CIRR/cirr/captions/cap.rc2.test12.json'
    dataset_path = Path('CIRR')

BLIP2_MODEL = 'opt' # or 'opt' or 't5'
MULTI_CAPTION = False
NUM_CAPTION = 1
with open(input_json, "r") as f:
    annotations = json.load(f)

for ans in tqdm(annotations):
    if DATASET == 'circo':
        rel_cap = ans["relative_caption"]
    elif DATASET == 'cirr':
        rel_cap = ans["caption"]
    if MULTI_CAPTION:
        blip2_caption = ans["multi_caption_{}".format(BLIP2_MODEL)]
    else:
        if BLIP2_MODEL == 'none':
            blip2_caption = ans["shared_concept"]
        else:
            blip2_caption = ans["blip2_caption_{}".format(BLIP2_MODEL)]

    sys_prompt = "I have an image. Given an instruction to edit the image, carefully generate a description of the edited image."

    if MULTI_CAPTION:
        multi_gpt = []
        ans["generated_query"] = list()
        for cap in blip2_caption:
            usr_prompt = "I will put my image content beginning with \"Image Content:\". The instruction I provide will begin with \"Instruction:\". The edited description you generate should begin with \"Edited Description:\". You just generate one edited description only begin with \"Edited Description:\". The edited description needs to be as simple as possible and only reflects image content. Just one line.\nA example:\nImage Content: a man adjusting a woman's tie.\nInstruction: has the woman and the man with the roles switched.\nEdited Description: a woman adjusting a man's tie.\n\nImage Content: {}\nInstruction: {}\nEdited Description:".format(cap, rel_cap)
        # print(prompt)
            ans["gemma_generated_query"].append(genertated_query(prompt = sys_prompt + '\n' + usr_prompt))
    else:
        usr_prompt = "I will put my image content beginning with \"Image Content:\". The instruction I provide will begin with \"Instruction:\". The edited description you generate should begin with \"Edited Description:\". You just generate one edited description only begin with \"Edited Description:\". The edited description needs to be as simple as possible and only reflects image content. Just one line.\nA example:\nImage Content: a man adjusting a woman's tie.\nInstruction: has the woman and the man with the roles switched.\nEdited Description: a woman adjusting a man's tie.\n\nImage Content: {}\nInstruction: {}\nEdited Description:".format(blip2_caption, rel_cap)
        # print(prompt)
        ans["gemma_generated_query"] = generated_query(prompt = sys_prompt + '\n' + usr_prompt)
           # with open("CIRCO/annotations/gpt3.5-temp.json", "a") as f:
    #     f.write(json.dumps(ans, indent=4) + '\n')

with open(input_json, "w") as f:
    f.write(json.dumps(annotations, indent=4))
