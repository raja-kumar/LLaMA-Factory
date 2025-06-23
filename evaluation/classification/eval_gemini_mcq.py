import json
import os
import time
from openai import OpenAI
import base64
from tqdm import tqdm
from typing import Dict, List
import argparse

def encode_image_to_base64(image_path: str) -> str:
    """Convert image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_gpt4v_response(client: OpenAI, image_path: str, prompt: str) -> str:
    """Get response from GPT-4V model."""
    try:
        base64_image = encode_image_to_base64(image_path)
        
        response = client.chat.completions.create(
            model="google/gemini-2.5-flash-lite-preview-06-17",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return ""

def evaluate_response(response: str, ground_truth: str) -> bool:
    """Evaluate if the model's response matches the ground truth."""
    try:
        # Extract answer from response if in <answer> format
        if "<answer>" in response and "</answer>" in response:
            predicted = response[response.find("<answer>")+8:response.find("</answer>")].strip()
        else:
            # Try to find just the letter answer
            for line in response.split('\n'):
                if len(line.strip()) == 1 and line.strip().upper() in "ABCDE":
                    predicted = line.strip()
                    break
            else:
                predicted = ""
        
        # Extract ground truth
        gt = ground_truth[ground_truth.find("<answer>")+8:ground_truth.find("</answer>")].strip()
        
        return predicted.upper() == gt.upper()
    except:
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the JSON dataset file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save results")
    args = parser.parse_args()

    # Initialize OpenAI client
    # client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    # Load dataset
    with open(args.data_path, 'r') as f:
        dataset = json.load(f)

    results = []
    correct = 0
    total = 0

    for item in tqdm(dataset):
        if "image_path" not in item or "problem" not in item or "solution" not in item:
            continue

        total += 1
        
        # Update image path to your local path if needed
        image_path = item["image_path"]
        image_path = image_path.replace("/home/raja/OVOD/git_files/VLM-COT/data/", "/app/shared_data/raja/")
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        # prompt = item["problem"].replace("Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.The output answer format should be as follows:\n<think> ... </think> <answer>option letter</answer>\nPlease strictly follow the format. ", "Output the final answer in <answer> </answer> tags. Strictly follow the format")

        response = get_gpt4v_response(client, image_path, item["problem"])
        is_correct = evaluate_response(response, item["solution"])
        
        if is_correct:
            correct += 1

        results.append({
            "image_path": image_path,
            "problem": item["problem"],
            "ground_truth": item["solution"],
            "model_response": response,
            "is_correct": is_correct
        })

        # Add delay to respect rate limits
        time.sleep(0.5)

    accuracy = (correct / total) * 100 if total > 0 else 0

    # Save results
    output = {
        "accuracy": accuracy,
        "total_samples": total,
        "correct_predictions": correct,
        "detailed_results": results
    }

    with open(args.output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Evaluation completed!")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Correct predictions: {correct}/{total}")
    print(f"Results saved to: {args.output_path}")

if __name__ == "__main__":
    main()