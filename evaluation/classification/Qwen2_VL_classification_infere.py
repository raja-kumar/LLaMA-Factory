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

def plot_images(image_paths):
    num_images = len(image_paths)
    
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
    
    for i, image_path in enumerate(image_paths):
        img = mpimg.imread(image_path)
        if num_images == 1:
            ax = axes
        else:
            ax = axes[i]
        ax.imshow(img)
        ax.set_title(f'Image {i+1}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


# ===== model path and model base =====

MODEL_ROOT = "/app/saved_models/vrft/CUB_200_2011/"  # root path for saved models
BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
EXP_NAME = "Qwen2_5-VL-7B-Instruct_GRPO_cub_base_1_shot_mcq"  # experiment name for saving models
CHECKPOINT = "checkpoint-400"  # checkpoint name for saved models


model_path = os.path.join(MODEL_ROOT, f"{EXP_NAME}", CHECKPOINT)  # full path to the model"
model_base = BASE_MODEL  # base model name

# ==== configurations ====

zero_shot = True
eval_type = "rft"  # "sft" or everything else
predict_top_5 = False  # top k for evaluation, default is 5
use_cat_list = False

if eval_type == "baseline":
    model_path = BASE_MODEL

# ==== dataset and output paths ====
DATA_ROOT = "/app/shared_data/raja/"
dataset = "CUB_200_2011"  # oxford_flowers, oxford-iiit-pet, CUB_200_2011
split = "base_val"  # split name, can be "base_train", "base_val", "new_test", "new_val" etc.

zero_shot_json_path = f"{DATA_ROOT}/{dataset}/zero_shot/subsample_{split}.json"

output_path = f"./output/{dataset}/{eval_type}/"

if "checkpoint" in model_path:
    model_name = model_path.split("/")[-2] + "_" + model_path.split("/")[-1] # use checkpoint name
else:
    model_name = model_path.split("/")[-1]  # model name

data_name = zero_shot_json_path.split("/")[-1].split(".")[0]  # data name
output_file = f"{model_name}_{data_name}_{use_cat_list}.json"  # output file name

if not os.path.exists(output_path):
    os.makedirs(output_path)

output_file_path = os.path.join(output_path, output_file)

print(GREEN + "output path" + output_file_path + RESET)
output_data = {}

### this is a temporary fix, will be removed later
one_shot_train_file = f"{DATA_ROOT}/{dataset}/fewshot/1_shots_base_train_mcq.json"
with open(one_shot_train_file, 'r') as f:
    one_shot_data = json.load(f)

def check_impath_in_training_data(image_path):
    """
    Check if the image path is in the one-shot training data.
    """
    image_id = image_path.split("/")[-1].split(".")[0]
    for item in one_shot_data:
        if item['image_path'].split("/")[-1].split(".")[0] == image_id:
            return True
    return False

### end of temporary fix

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

    with open(zero_shot_json_path, 'r') as f:
        infer_data = json.load(f)
    
    random.seed(21)
    random.shuffle(infer_data)

    # infer_data = infer_data[:10]

    print(GREEN + "Number of images in infer data: " + str(len(infer_data)) + RESET)
    

    rank = rank
    world_size = world_size
    import math
    split_length = math.ceil(len(infer_data)/world_size)
    logger.info("Split Chunk Length:" + str(split_length))
    split_images = infer_data[int(rank*split_length) : int((rank+1)*split_length)]
    logger.info(len(split_images))

    '''
    To do:
        - Load the categories correctly. Add categories list to the question if use_cat_list is True. 
    '''

    categories = []
    

    error_count = 0
    right_count = 0
    for item in tqdm(split_images, total=len(split_images), desc=f"Rank {rank} Processing"):
        image_path = item['image_path']
        image_label = item['solution']

        ### temporary fix for one-shot training data
        if check_impath_in_training_data(image_path):
            logger.info(f"Skipping image {image_path} as it is in the one-shot training data.")
            continue

        ### end of temporary fix
        prompt = item['problem']
        image_label = re.search(r"<answer>(.*?)</answer>", image_label).group(1)
        image_path = image_path.replace("/home/raja/OVOD/git_files/VLM-COT/data/", 
                        DATA_ROOT)


        if predict_top_5:
            temp = "output the top five most likely species names in the image. Even if you are sure about the answer, output top 5 categories."
            answer_format = "[category 1, category 2, catefory 3, category 4, category 5]"
        else:
            temp = "output the most likely species name in the image."
            answer_format = "species name"

        if use_cat_list:
            question = (
            f"This is an image containing a flower. {temp}\n"
            f"the species of the plant strictly belongs to below category list {categories}.\n"
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
            "The output answer format should be as follows:\n"
            f"<think> ... </think> <answer>{answer_format}</answer>\n"
            "Please strictly follow the format."
            )
        else:
            question = prompt
        # print(RED + question + RESET)
    
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
                    "groundtruth": image_label,
                    "reasoning": "", # No reasoning for SFT
                    "answer": response
                }

                image_label = image_label.replace(' ','').replace('_','').lower()
                response_lower = response.replace(' ','').replace('_','').lower()

                if image_label in response_lower:
                    right_count += 1
                else:
                    error_count += 1
            else:
                # For other cases, keep the original parsing logic
                reasoning = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
                reasoning_content = reasoning.group(1).strip() if reasoning else ""
                match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
                if not match:
                    match = re.search(r"<answer>\n(.*?)</answer>", response, re.DOTALL)
                if not match:
                    match = re.search(r"<answer>\n(.*?)\n</answer>", response, re.DOTALL)
                
                answer_content = match.group(1)

                image_id = image_path.split("/")[-1].split(".")[0]

                local_output_data[image_id] = {
                    "groundtruth": image_label,
                    "reasoning": reasoning_content,
                    "answer": answer_content
                }

                # print(local_output_data[image_id])
                if ("describe" in model_path):
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
        except Exception as e:
            print(RED + "Error in processing response: " + response + RESET)
            error_count += 1
        
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
        logger.info("above count holds meaning only for sft eval. IGNORE for other evals.")
    else:
        logger.info("Not enough GPUs")

if __name__ == "__main__":
    main()

    with open(output_file_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    logger.info(f"Output saved to {output_file_path}")