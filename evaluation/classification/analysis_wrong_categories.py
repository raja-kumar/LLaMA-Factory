import json
from collections import defaultdict


input_data_path = "/data2/raja/oxford_flowers/zero_shot_mcq/subsample_base_train.json"
# output_data_path = "./output/rft_mcq/Qwen2.5-VL-7B-Instruct_subsample_base_val.json"
# output_data_path = "./output/rft_mcq/Qwen2_5-VL-7B-Instruct_GRPO_flowers_base_mcq_subsample_base_val.json"
output_data_path = "./output/rft_mcq/Qwen2_5-VL-7B-Instruct_GRPO_flowers_base_mcq_checkpoint-300_subsample_base_train.json"

hard_data_path = "/data2/raja/oxford_flowers/zero_shot_mcq/hard_subsample_base_train.json"

# Function to extract category name from problem text given the correct option letter
def get_category_from_problem(problem_text, answer_letter):
    # Split the problem text by newlines and find the line starting with the answer letter
    lines = problem_text.split('\n')
    for line in lines:
        if line.startswith(f"{answer_letter}."):
            return line.split('. ')[1].strip()
    return None

# Read input data
with open(input_data_path, 'r') as f:
    input_data = json.load(f)

# Read output data 
with open(output_data_path, 'r') as f:
    output_data = json.load(f)

# Create dictionaries to store error counts and total counts per category
error_counts = defaultdict(int)
total_counts = defaultdict(int)

hard_data = []

# Analyze each example
for input_item in input_data:
    # Extract image name from path
    image_name = input_item['image_path'].split('/')[-1].replace('.jpg', '')
    
    if image_name in output_data:
        output_item = output_data[image_name]
        
        # Get ground truth category
        ground_truth_letter = output_item['groundtruth']
        ground_truth_category = get_category_from_problem(input_item['problem'], ground_truth_letter)
        
        if ground_truth_category:
            # Increment total count for this category
            total_counts[ground_truth_category] += 1
            
            # Check if model made a mistake
            if output_item['answer'] != output_item['groundtruth']:
                error_counts[ground_truth_category] += 1
                # Store the hard example
                hard_data.append(input_item)

# Save hard examples to a file
with open(hard_data_path, 'w') as f:
    json.dump(hard_data, f, indent=4)

# Calculate and display error rates per category
print("\nError distribution by category:")
print("-" * 50)
print(f"{'Category':<30} {'Errors':<8} {'Total':<8} {'Error Rate':<10}")
print("-" * 50)

for category in sorted(total_counts.keys()):
    errors = error_counts[category]
    total = total_counts[category]
    error_rate = (errors / total) * 100 if total > 0 else 0
    
    print(f"{category:<30} {errors:<8} {total:<8} {error_rate:.2f}%")

# Overall statistics
total_errors = sum(error_counts.values())
total_samples = sum(total_counts.values())
overall_error_rate = (total_errors / total_samples) * 100 if total_samples > 0 else 0

print("\nOverall Statistics:")
print(f"Total errors: {total_errors}")
print(f"Total samples: {total_samples}")
print(f"Overall error rate: {overall_error_rate:.2f}%")

# Find categories with highest error rates
print("\nCategories with highest error rates:")
sorted_categories = sorted(
    [(cat, error_counts[cat], total_counts[cat]) for cat in total_counts],
    key=lambda x: (x[1]/x[2] if x[2]>0 else 0),
    reverse=True
)

for cat, errors, total in sorted_categories[:5]:
    error_rate = (errors / total) * 100 if total > 0 else 0
    print(f"{cat}: {error_rate:.2f}% ({errors}/{total})")