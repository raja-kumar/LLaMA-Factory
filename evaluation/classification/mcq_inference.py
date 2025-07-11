import io
import os
import re
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          StoppingCriteria, StoppingCriteriaList)
from transformers.generation import GenerationConfig
from peft import AutoPeftModelForCausalLM
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
torch.manual_seed(1234)

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from datetime import datetime

# 定义颜色的ANSI代码
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'  # 重置颜色


import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import functools
import itertools
import multiprocessing as mp
from argparse import ArgumentParser
from multiprocessing import Pool

import random
random.seed(21)
# from utils import get_cat_name_from_json


def extract_choice(text):
    # 1. Clean and normalize text
    text = text.upper()  # Convert to uppercase
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces

    # 2. Choice should not have uppercase letters before or after
    choices = re.findall(r'(?<![A-Z])([A-Z])(?=[\.\,\?\!\:\;]|$)', text)

    if not choices:
        return None

    # 3. If only one choice, return it directly
    if len(choices) == 1:
        return choices[0]

    # 4. If multiple choices, use heuristic rules
    choice_scores = {choice: 0 for choice in choices}

    # 4.1 Keywords around choices get points
    keywords = [
        '答案', '选择', '正确', '是', '对',
        'answer', 'correct', 'choose', 'select', 'right',
        '认为', '应该', '觉得', 'think', 'believe', 'should'
    ]

    # Get context for each choice (20 chars before and after)
    for choice in choices:
        pos = text.find(choice)
        context = text[max(0, pos-20):min(len(text), pos+20)]

        # Add points for keywords
        for keyword in keywords:
            if keyword.upper() in context:
                choice_scores[choice] += 1

        # Add points if choice is near the end (usually final answer)
        if pos > len(text) * 0.7:  # In last 30% of text
            choice_scores[choice] += 2

        # Add points if followed by punctuation
        if pos < len(text) - 1 and text[pos+1] in '。.!！,，':
            choice_scores[choice] += 1

    # Return highest scoring choice
    return max(choice_scores.items(), key=lambda x: x[1])[0]



# model path and model base
# model_path = "/app/saves/flowers_4_shot/qwen2_vl-2b/full/sft/checkpoint-306/"  # after SFT
# model_path = "Qwen/Qwen2-VL-2B-Instruct"
# model_path = "/app/saved_models/LLaMA-Factory/saves/flowers_base/qwen2_vl-2b/full/sft/checkpoint-1308"  # after SFT
# model_path = "/app/saved_models/vrft/ckpts/Qwen2-VL-2B-Instruct_GRPO_flowers_base/checkpoint-1308"
# model_path = "/app/saved_models/vrft/ckpts/Qwen2-VL-2B-Instruct_GRPO_flowers_base_updated_reward/checkpoint-1308"
# model_path = "/app/saved_models/vrft/ckpts/Qwen2-VL-2B-Instruct_GRPO_flowers_base_mcq/checkpoint-1302"
# model_path = "/app/saved_models/vrft/ckpts/Qwen2-VL-2B-Instruct_GRPO_flowers_base_mcq/checkpoint-400"  # after GRPO
# model_base = "Qwen/Qwen2-VL-2B-Instruct"  # original Qwen2-VL

## Qwen2.5

# model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
# model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
# model_path = "/app/saves/flowers_4_shot/qwen2_5_vl_3b/full/sft/checkpoint-306"  # after SFT
# model_path = "/app/saved_models/vrft/ckpts/Qwen2_5-VL-7B-Instruct_GRPO_flowers_base_mcq/checkpoint-500"
# model_path = "/app/saved_models/vrft/ckpts/Qwen2_5-VL-7B-Instruct_GRPO_flowers_base_updated_reward/checkpoint-291"
# model_path = "/app/saved_models/vrft/ckpts/Qwen2_5-VL-7B-Instruct_GRPO_flowers_base_mcq/checkpoint-300"
# model_path = "/app/saved_models/vrft/ckpts/Qwen2_5-VL-7B-Instruct_GRPO_flowers_base_hard_examples/checkpoint-200"
# model_path = "/app/saved_models/vrft/ckpts/Qwen2_5-VL-7B-Instruct_GRPO_flowers_base_mcq_describe/checkpoint-870"
# model_path = "/app/saved_models/vrft/ckpts/Qwen2_5-VL-7B-Instruct_GRPO_flowers_base_4_shot_describe/checkpoint-400"
# model_path = "/app/saved_models/vrft/ckpts/Qwen2_5-VL-7B-Instruct_GRPO_flowers_base_4_shot_and_hard/checkpoint-400"
# model_path = "/app/saved_models/vrft/ckpts/Qwen2_5-VL-7B-Instruct_GRPO_flowers_base_mcq/checkpoint-300"
model_path = "/app/saved_models/vrft/ckpts/Qwen2_5-VL-7B-Instruct_GRPO_flowers_base_and_hard_mcq/checkpoint-400"
model_base = "Qwen/Qwen2.5-VL-7B-Instruct"
# categories_json = "../data/oxford_flowers/idx_2_class.json"  # categories json file


# ==== configurations ====

zero_shot = True
eval_type = "rft_mcq"  # "sft" or everything else
predict_top_5 = False  # top k for evaluation, default is 5

zero_shot_json_path = "/app/shared_data/raja/oxford_flowers/zero_shot_mcq/subsample_base_val.json"
# zero_shot_json_path = "/app/shared_data/raja/oxford_flowers/zero_shot_mcq/subsample_new_test.json"
# zero_shot_json_path = "/app/shared_data/raja/oxford_flowers/zero_shot_mcq/subsample_base_train.json"
# zero_shot_json_path = "/app/shared_data/raja/oxford_flowers/zero_shot_mcq/hard_subsample_base_train.json"
# zero_shot_json_path = "/app/shared_data/raja/oxford_flowers/zero_shot_mcq/subsample_base_val_describe.json"
# zero_shot_json_path = "/app/shared_data/raja/oxford_flowers/zero_shot_mcq/subsample_new_test_describe.json"
# zero_shot_json_path = "/app/shared_data/raja/oxford_pet/zero_shot/subsample_base_val_mcq.json"
# zero_shot_json_path = "/app/shared_data/raja/oxford-iiit-pet/zero_shot/subsample_new_val_mcq.json"

## dataset detail
# dataset = "oxford_pet"  # dataset name, used for output path
dataset = "oxford_flowers"  # dataset name, used for output path

output_path = f"./output/{dataset}/{eval_type}/"

if "checkpoint" in model_path:
    model_name = model_path.split("/")[-2] + "_" + model_path.split("/")[-1]  # use checkpoint name
else:
    model_name = model_path.split("/")[-1]  # model name

data_name = zero_shot_json_path.split("/")[-1].split(".")[0]  # data name
output_file = f"{model_name}_{data_name}.json"  # output file name

if not os.path.exists(output_path):
    os.makedirs(output_path)

output_file_path = os.path.join(output_path, output_file)

if os.path.exists(output_file_path):
    user_input = input(f"The file '{output_file_path}' already exists. Do you want to overwrite it? (yes/no): ").strip().lower()
    if user_input not in ['yes', 'y']:
        print("Operation aborted by the user.")
        exit(0)  # Exit the script if the user does not confirm

print(GREEN + "output path " + output_file_path + RESET)
output_data = {}

def run(rank, world_size):

    local_output_data = {}

    if "Qwen2.5" in model_base:

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cpu",
        )
    
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cpu",
        )

    processor = AutoProcessor.from_pretrained(model_base) 

    model = model.to(torch.device(rank))
    model = model.eval()

    ### get categories name
    with open('./val_data/oxford_flowers.txt', 'r') as file:
        lines = file.readlines()
    categories = []
    for line in lines:
        categories.append(line.strip())
    # print(len(categories))
    # print(categories)   ### 对应 0-101

    val_set = []

    

    with open(zero_shot_json_path, 'r') as f:
        predictions = json.load(f)
    
    random.seed(21)
    random.shuffle(predictions)

    print(len(predictions))

    # predictions = predictions[:5]  # limit to 1000 for testing
    rank = rank
    world_size = world_size
    import math
    split_length = math.ceil(len(predictions)/world_size)
    logger.info("Split Chunk Length:" + str(split_length))
    split_images = predictions[int(rank*split_length) : int((rank+1)*split_length)]
    logger.info(len(split_images))

    error_count = 0
    right_count = 0

    for item in tqdm(split_images):
        image_path = item['image_path']
        if (not os.path.exists(image_path)):
            image_path = image_path.replace("/home/raja/OVOD/git_files/VLM-COT/data/", "/app/shared_data/raja/")
        image_prompt = item["problem"]
        image_label = item['solution']
        # image_label = re.search(r"<answer>(.*?)</answer>", image_label).group(1)
        image_path = image_path.replace("/home/raja/OVOD/git_files/VLM-COT/data/oxford_flowers/jpg/", 
                        "/app/shared_data/raja/oxford_flowers/jpg/")
        # val_set.append({image_path: image_label})
        
        if (not zero_shot):
            image_cate = categories[image_cate]   

        # print(RED + question + RESET)

        question = image_prompt
    
        image_path = image_path
        query = "<image>\n"+question
        # print(RED+query+RESET)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path}
                ] + [{"type": "text", "text": query}],
            }
        ]
        
        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        
        # Inference: Generation of the output
        if predict_top_5:
            generated_ids = model.generate(**inputs, max_new_tokens=1024, use_cache=True, temperature=1.1, do_sample=True)
        else:
            generated_ids = model.generate(**inputs, max_new_tokens=1024, use_cache=True)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = response[0]
        
        # print("\033[92m" + response + "\033[0m")

        try:
            if eval_type == "sft":
                # For SFT, search in complete response without parsing

                image_id = image_path.split("/")[-1].split(".")[0]

                local_output_data[image_id] = {
                    "groundtruth": image_cate,
                    "reasoning": "", # No reasoning for SFT
                    "answer": response
                }

                image_cate = image_cate.replace(' ','').replace('_','').lower()
                response_lower = response.replace(' ','').replace('_','').lower()

                if image_cate in response_lower:
                    right_count += 1
                else:
                    error_count += 1
            else:
                # print(YELLOW + "Processing response: " + response + RESET)
                # print(GREEN + "Image Label: " + image_label + RESET)
                sol_match = re.search(r'<answer>(.*?)</answer>', image_label)
                # print(GREEN + "Solution Match: " + str(sol_match) + RESET)
                ground_truth = sol_match.group(1).strip() if sol_match else image_label.strip()
                has_choices = extract_choice(ground_truth)
                # print(GREEN + "Has Choices: " + str(has_choices) + RESET)
                correct_choice = has_choices.upper() if has_choices else image_label.strip()
                # print(GREEN + "Correct Choice: " + correct_choice + RESET)

                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                # print(GREEN + "Content Match: " + str(content_match) + RESET)
                student_answer = content_match.group(1).strip() if content_match else response.strip()
                student_choice = extract_choice(student_answer)
                # print(GREEN + "Student Choice: " + str(student_choice) + RESET)
                if student_choice:
                    if student_choice == correct_choice:
                        right_count += 1
                    else:
                        error_count += 1
                else:
                    error_count += 1

                reasoning = re.search(r"<think>(.*?)</think>", response)
                # print(GREEN + "Reasoning Match: " + str(reasoning) + RESET)
                reasoning_content = reasoning.group(1) if reasoning else ""
                # print(GREEN + "Reasoning Content: " + reasoning_content + RESET)
                image_id = image_path.split("/")[-1].split(".")[0]
                local_output_data[image_id] = {
                    "groundtruth": correct_choice,
                    "reasoning": reasoning_content,
                    "answer": student_choice
                }
                if ("describe" in zero_shot_json_path or "describe" in model_path):
                    # For describe task, we use the image_id as the key
                    describe_match = re.search(r'<describe>(.*?)</describe>', response, re.DOTALL)
                    if describe_match:
                        describe_content = describe_match.group(1).strip()
                    else:
                        describe_content = ""
                    
                    rethink_match = re.search(r'<rethink>(.*?)</rethink>', response, re.DOTALL)
                    if rethink_match:
                        rethink_content = rethink_match.group(1).strip()
                    else:
                        rethink_content = ""
                    
                    local_output_data[image_id]["describe"] = describe_content
                    local_output_data[image_id]["rethink"] = rethink_content

                

                # print(GREEN + image_id + "local_output_data: " + str(local_output_data[image_id]) + RESET)
        except Exception as e:
            print(RED + "Error in processing response: " + response + RESET)
            error_count += 1
        
    # print(output_data)        
    return [error_count, right_count, local_output_data]

def main():
    multiprocess = torch.cuda.device_count() >= 1
    mp.set_start_method('spawn')
    if multiprocess:
        logger.info('started generation')
        n_gpus = torch.cuda.device_count()
        world_size = n_gpus
        with Pool(world_size) as pool:
            func = functools.partial(run, world_size=world_size)
            result_lists = pool.map(func, range(world_size))

        global_count_error = 0
        global_count_right = 0
        global_results = []
        for i in range(world_size):
            logger.info('Rank: ' + str(i) + ' Error Number: ' + str(result_lists[i][0]) + 
                        ' Right Number: ' + str(result_lists[i][1]))
            global_count_error += int(result_lists[i][0])
            global_count_right = global_count_right + result_lists[i][1]

            output_data.update(result_lists[i][2])  # merge local output data
            
        logger.info('Error number: ' + str(global_count_error))  
        logger.info('Total Right Number: ' + str(global_count_right))
    else:
        logger.info("Not enough GPUs")

if __name__ == "__main__":
    main()

    with open(output_file_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    logger.info(f"Output saved to {output_file_path}")