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




# model path and model base
# model_path = "/app/saves/flowers_4_shot/qwen2_vl-2b/full/sft/checkpoint-306/"  # after RL
model_path = "Qwen/Qwen2-VL-2B-Instruct"
# model_path = "/app/saves/flowers_base/qwen2_vl-2b/full/sft/checkpoint-1308/"  # after SFT
model_base = "Qwen/Qwen2-VL-2B-Instruct"  # original Qwen2-VL

## Qwen2.5

# model_path = "Qwen/Qwen2.5-VL-3B-Instruct"  
# model_path = "/app/saves/flowers_4_shot/qwen2_5_vl_3b/full/sft/checkpoint-306/"  # after SFT
# model_base = "Qwen/Qwen2.5-VL-3B-Instruct"  

# categories_json = "../data/oxford_flowers/idx_2_class.json"  # categories json file

def run(rank, world_size):

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

    ## set these flags
    use_cat_list = True
    zero_shot = True
    eval_type = "baseline"  # "sft" or "baseline" or rft

    ### get categories name
    with open('./val_data/oxford_flowers.txt', 'r') as file:
        lines = file.readlines()
    categories = []
    for line in lines:
        categories.append(line.strip())
    # print(len(categories))
    # print(categories)   ### 对应 0-101

    val_set = []
    zero_shot_json_path = "/app/shared_data/raja/oxford_flowers/zero_shot/subsample_new_test.json"

    
    if zero_shot:
        with open(zero_shot_json_path, 'r') as f:
            predictions = json.load(f)
        
        for item in predictions:
            image_path = item['image_path']
            image_label = item['solution']
            image_label = re.search(r"<answer>(.*?)</answer>", image_label).group(1)
            image_path = image_path.replace("/home/raja/OVOD/git_files/VLM-COT/data/oxford_flowers/jpg/", 
                            "/app/shared_data/raja/oxford_flowers/jpg/")
            val_set.append({image_path: image_label})
    else:
        ### get validation data
        pth_file_path = './val_data/oxford_flowers.pth'
        predictions = torch.load(pth_file_path)

        for item in predictions:
            for k,v in item.items():
                k = k.replace("/mnt/petrelfs/liuziyu/LLM_Memory/SimplyRetrieve/CLIP-Cls/data/oxford_flowers/jpg/", 
                                "/app/shared_data/raja/oxford_flowers/jpg/")
                val_set.append({k:int(v['label'])})
    
    print(len(val_set))
    # print(val_set[0])

    # val_set = val_set[:5]  # for test

    rank = rank
    world_size = world_size
    import math
    split_length = math.ceil(len(val_set)/world_size)
    logger.info("Split Chunk Length:" + str(split_length))
    split_images = val_set[int(rank*split_length) : int((rank+1)*split_length)]
    logger.info(len(split_images))

    ### 遍历 val 中的所有图片
    error_count = 0
    right_count = 0
    for image in tqdm(split_images): 
        ### 获取图片信息
        for k,v in image.items():
            image_path = k
            image_cate = v
        
        if (not zero_shot):
            image_cate = categories[image_cate]   
        # plot_images([image_path])

        if use_cat_list:
            question = (
            "This is an image containing a plant. Please identify the species of the plant based on the image.\n"
            f"the species of the plant belongs to below category list {categories}.\n"
            "answer strictly from the category list.\n"
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
            "The output answer format should be as follows:\n"
            "<think> ... </think> <answer>species name</answer>\n"
            "Please strictly follow the format."
            )
        else:
            question = (
            "This is an image containing a plant. Please identify the species of the plant based on the image.\n"
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
            "The output answer format should be as follows:\n"
            "<think> ... </think> <answer>species name</answer>\n"
            "Please strictly follow the format."
            )

        # print(RED + question + RESET)
    
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
                image_cate = image_cate.replace(' ','').replace('_','').lower()
                response_lower = response.replace(' ','').replace('_','').lower()
                # judgement
                # print("response_lower: ", response_lower)
                # print("image_cate: ", image_cate)
                if image_cate in response_lower:
                    right_count += 1
                else:
                    error_count += 1
            else:
                # For other cases, keep the original parsing logic
                match = re.search(r"<answer>(.*?)</answer>", response)
                answer_content = match.group(1)
                image_cate = image_cate.replace(' ','').replace('_','').lower()
                answer_content = answer_content.replace(' ','').replace('_','').lower()
                # judgement
                if image_cate in answer_content or answer_content in image_cate:
                    right_count += 1
                else:
                    error_count += 1
        except Exception as e:
            error_count += 1
        
        # try:
        #     match = re.search(r"<answer>(.*?)</answer>", response)
        #     answer_content = match.group(1)
        #     # print(image_cate, answer_content)
        #     image_cate = image_cate.replace(' ','').replace('_','').lower()
        #     answer_content = answer_content.replace(' ','').replace('_','').lower()
        #     # judgement
        #     if image_cate in answer_content or answer_content in image_cate or image_cate in response:
        #         # print('yes')
        #         right_count += 1
        #         # logger.info('Local Right Number: ' + str(right_count))
        #     else:
        #         # print('no')
        #         error_count+=1
        # except Exception as e:
        #     error_count+=1
            
    return [error_count, right_count]

def main():
    multiprocess = torch.cuda.device_count() >= 2
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
            
        logger.info('Error number: ' + str(global_count_error))  
        logger.info('Total Right Number: ' + str(global_count_right))
    else:
        logger.info("Not enough GPUs")

if __name__ == "__main__":
    main()