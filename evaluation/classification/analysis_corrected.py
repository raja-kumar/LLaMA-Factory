import json

# baseline = "./output/baseline/Qwen2-VL-2B-Instruct_subsample_base_val_False_evaluation.json"
# rft = "./output/rft/Qwen2-VL-2B-Instruct_GRPO_flowers_base_updated_reward_subsample_base_val_False_evaluation.json"
# Load both JSON files

# rft = "/home/raja/gdrive/LLaMA-Factory/evaluation/classification/output/rft/Qwen2_5-VL-7B-Instruct_GRPO_flowers_base_mcq_checkpoint-300_subsample_new_test_False_evaluation.json"
# baseline = "/home/raja/gdrive/LLaMA-Factory/evaluation/classification/output/baseline/Qwen2.5-VL-7B-Instruct_subsample_new_test_False_evaluation.json"
# baseline = "/home/raja/gdrive/LLaMA-Factory/evaluation/classification/output/rft/Qwen2_5-VL-7B-Instruct_GRPO_flowers_base_updated_reward_checkpoint-291_subsample_new_test_False_evaluation.json"
baseline = "./output/rft_mcq/Qwen2_5-VL-7B-Instruct_GRPO_flowers_base_mcq_describe_checkpoint-870_subsample_base_val_False_evaluation.json"
rft = "./output/rft/Qwen2_5-VL-7B-Instruct_GRPO_flowers_base_mcq_checkpoint-300_subsample_base_val_False_evaluation.json"

with open(baseline, 'r') as f1, open(rft, 'r') as f2:
    data1 = json.load(f1)["detailed_results"]
    data2 = json.load(f2)["detailed_results"]

# Find image IDs where top1_correct is false in json1 and true in json2
improved_ids = []
for image_id in data1:
    if image_id in data2:
        if data1[image_id]["top1_correct"] == False and data2[image_id]["top1_correct"] == True:
            improved_ids.append(image_id)

# Output the results
print("Images where top1_correct improved from false to true:")
for img_id in improved_ids:
    print(img_id)
print(f"Total images with improvement: {len(improved_ids)}")
