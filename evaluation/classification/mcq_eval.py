import json

def calculate_accuracy(json_file_path):
    # Read the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    total = len(data)
    correct = 0

    no_reasoning = 0
    correct_with_no_reasoning = 0
    
    # Compare groundtruth with answer for each image
    for image_id, values in data.items():
        if values['groundtruth'] == values['answer']:
            correct += 1
        else:
            print(f"Image ID: {image_id}, Groundtruth: {values['groundtruth']}, Answer: {values['answer']}")
        
        if (values["reasoning"] == ""):
            print(f"Image ID: {image_id}, Reasoning: {values['reasoning']}")
            no_reasoning += 1
            if (values['groundtruth'] == values['answer']):
                correct_with_no_reasoning += 1
    
    # Calculate accuracy
    accuracy = (correct / total) * 100
    
    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Samples with no reasoning: {no_reasoning}")
    print(f"Correct with no reasoning: {correct_with_no_reasoning}")
    
    return accuracy

# Use the function
json_file_path = "./output/rft_mcq/Qwen2_5-VL-7B-Instruct_GRPO_flowers_base_hard_examples_checkpoint-200_subsample_new_test.json"
calculate_accuracy(json_file_path)