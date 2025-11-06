#!/usr/bin/env python3
"""
Convert downloaded datasets to Prometheus instruction-output format
"""

import json
from pathlib import Path
from typing import List, Dict
import random

print("="*70)
print("Converting Datasets to Prometheus Format")
print("="*70)

data_dir = Path("C:/Users/Prithvi Putta/prometheus/training_data")
output_dir = Path("C:/Users/Prithvi Putta/prometheus/prometheus_formatted_data")
output_dir.mkdir(exist_ok=True)

# Prometheus format template
PROMETHEUS_TEMPLATE = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
[{criteria}]
Score 1: {score_1}
Score 2: {score_2}
Score 3: {score_3}
Score 4: {score_4}
Score 5: {score_5}

###Feedback: """


def convert_feedback_collection(input_path: Path, output_path: Path, max_samples: int = 10000):
    """
    Convert Feedback Collection dataset (already in Prometheus format)
    """
    print(f"\n[1/3] Converting Feedback Collection...")
    print(f"  Input: {input_path}")

    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    print(f"  Loaded: {len(data)} examples")

    # This dataset is already in Prometheus format, just need to reformat
    converted = []
    for item in data[:max_samples]:
        # Check if it has the expected fields
        if 'orig_instruction' in item and 'orig_response' in item:
            # Build Prometheus format
            instruction_text = PROMETHEUS_TEMPLATE.format(
                instruction=item.get('orig_instruction', ''),
                response=item.get('orig_response', ''),
                reference_answer=item.get('orig_reference_answer', 'N/A'),
                criteria=item.get('orig_criteria', 'Overall Quality'),
                score_1=item.get('orig_score1_description', 'Poor quality'),
                score_2=item.get('orig_score2_description', 'Below average'),
                score_3=item.get('orig_score3_description', 'Average'),
                score_4=item.get('orig_score4_description', 'Good'),
                score_5=item.get('orig_score5_description', 'Excellent')
            )

            converted.append({
                "instruction": instruction_text,
                "input": "",
                "output": item.get('orig_feedback', 'No feedback') + f" [RESULT] {item.get('orig_score', 3)}"
            })

    print(f"  Converted: {len(converted)} examples")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)

    print(f"  Saved to: {output_path}")
    return len(converted)


def convert_ultrafeedback(input_path: Path, output_path: Path, max_samples: int = 5000):
    """
    Convert UltraFeedback dataset to Prometheus format
    """
    print(f"\n[2/3] Converting UltraFeedback...")
    print(f"  Input: {input_path}")

    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    print(f"  Loaded: {len(data)} examples")

    converted = []
    for item in data[:max_samples]:
        # UltraFeedback has instruction and multiple completions with ratings
        instruction = item.get('instruction', '')
        completions = item.get('completions', [])

        if not completions:
            continue

        # Find the best completion (highest overall_score)
        best_completion = max(completions, key=lambda x: x.get('overall_score', 0))
        best_score = best_completion.get('overall_score', 3)

        # Normalize score to 1-5 if needed
        if best_score > 5:
            best_score = min(5, int(best_score / 2))

        # Use first completion as response to evaluate, best as reference
        response_completion = completions[0] if len(completions) > 0 else best_completion
        response = response_completion.get('response', '')
        response_score = response_completion.get('overall_score', 3)

        if response_score > 5:
            response_score = min(5, int(response_score / 2))

        reference_answer = best_completion.get('response', 'N/A')

        # Generate feedback based on scores
        feedback_parts = []
        for aspect in ['helpfulness', 'honesty', 'instruction_following', 'truthfulness']:
            if aspect in response_completion:
                rating = response_completion[aspect].get('Rating', 'N/A')
                rationale = response_completion[aspect].get('Rationale', '')
                if rationale:
                    feedback_parts.append(f"{aspect.title()}: {rationale[:200]}")

        feedback = " ".join(feedback_parts) if feedback_parts else "The response addresses the instruction."

        instruction_text = PROMETHEUS_TEMPLATE.format(
            instruction=instruction,
            response=response,
            reference_answer=reference_answer[:500],  # Limit length
            criteria="Overall Quality (Helpfulness, Honesty, Instruction Following, Truthfulness)",
            score_1="Poor: Unhelpful, dishonest, or does not follow instructions",
            score_2="Below Average: Some issues with helpfulness, honesty, or instruction following",
            score_3="Average: Adequately helpful and follows instructions",
            score_4="Good: Helpful, honest, and follows instructions well",
            score_5="Excellent: Highly helpful, completely honest, perfectly follows instructions"
        )

        converted.append({
            "instruction": instruction_text,
            "input": "",
            "output": f"{feedback} [RESULT] {response_score}"
        })

    print(f"  Converted: {len(converted)} examples")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)

    print(f"  Saved to: {output_path}")
    return len(converted)


def convert_hh_rlhf(input_path: Path, output_path: Path, max_samples: int = 3000):
    """
    Convert HH-RLHF dataset to Prometheus format
    """
    print(f"\n[3/3] Converting HH-RLHF...")
    print(f"  Input: {input_path}")

    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    print(f"  Loaded: {len(data)} examples")

    converted = []
    for item in data[:max_samples]:
        # HH-RLHF has 'chosen' (better) and 'rejected' (worse) responses
        chosen = item.get('chosen', '')
        rejected = item.get('rejected', '')

        if not chosen or not rejected:
            continue

        # Extract the last human message as instruction
        # Format is typically: "\n\nHuman: {text}\n\nAssistant: {response}"
        parts = chosen.split('\n\nHuman: ')
        if len(parts) < 2:
            continue

        # Get the instruction (last human message)
        human_parts = parts[-1].split('\n\nAssistant: ')
        if len(human_parts) < 2:
            continue

        instruction = human_parts[0].strip()
        chosen_response = human_parts[1].strip()

        # Get rejected response
        rejected_parts = rejected.split('\n\nAssistant: ')
        if len(rejected_parts) < 2:
            continue
        rejected_response = rejected_parts[-1].strip()

        # Create two training examples: one for chosen (score 5), one for rejected (score 2)
        for response, score, label in [
            (chosen_response, 5, "chosen"),
            (rejected_response, 2, "rejected")
        ]:
            instruction_text = PROMETHEUS_TEMPLATE.format(
                instruction=instruction[:500],
                response=response[:500],
                reference_answer=chosen_response[:500] if label == "rejected" else "An ideal, helpful, and harmless response",
                criteria="Helpfulness and Harmlessness",
                score_1="Harmful, unhelpful, or inappropriate",
                score_2="Somewhat unhelpful or potentially problematic",
                score_3="Neutral: neither particularly helpful nor harmful",
                score_4="Helpful and harmless with minor issues",
                score_5="Very helpful, completely harmless, and appropriate"
            )

            feedback = f"This response is the {label} one. " + (
                "It demonstrates helpfulness and safety." if label == "chosen"
                else "It has issues with helpfulness or safety compared to better alternatives."
            )

            converted.append({
                "instruction": instruction_text,
                "input": "",
                "output": f"{feedback} [RESULT] {score}"
            })

    print(f"  Converted: {len(converted)} examples")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)

    print(f"  Saved to: {output_path}")
    return len(converted)


def create_combined_dataset(output_paths: List[Path], combined_path: Path, shuffle: bool = True):
    """Combine all datasets into one training file"""
    print(f"\n[Combining] Creating unified training dataset...")

    all_data = []
    for path in output_paths:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_data.extend(data)
            print(f"  Added {len(data)} from {path.name}")

    if shuffle:
        random.shuffle(all_data)
        print(f"  Shuffled {len(all_data)} total examples")

    # Split into train and validation
    split_idx = int(len(all_data) * 0.9)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]

    train_path = combined_path.parent / "prometheus_train.json"
    val_path = combined_path.parent / "prometheus_val.json"

    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)

    print(f"  Training set: {len(train_data)} examples -> {train_path}")
    print(f"  Validation set: {len(val_data)} examples -> {val_path}")

    return train_path, val_path


# Main conversion
print("\nStarting dataset conversion...\n")

output_paths = []
total_samples = 0

# Convert each dataset
if (data_dir / "feedback_collection.json").exists():
    out_path = output_dir / "feedback_collection_formatted.json"
    count = convert_feedback_collection(data_dir / "feedback_collection.json", out_path)
    output_paths.append(out_path)
    total_samples += count

if (data_dir / "ultrafeedback.json").exists():
    out_path = output_dir / "ultrafeedback_formatted.json"
    count = convert_ultrafeedback(data_dir / "ultrafeedback.json", out_path)
    output_paths.append(out_path)
    total_samples += count

if (data_dir / "hh_rlhf.json").exists():
    out_path = output_dir / "hh_rlhf_formatted.json"
    count = convert_hh_rlhf(data_dir / "hh_rlhf.json", out_path)
    output_paths.append(out_path)
    total_samples += count

# Create combined dataset
combined_path = output_dir / "prometheus_combined.json"
train_path, val_path = create_combined_dataset(output_paths, combined_path)

print("\n" + "="*70)
print("Conversion Complete!")
print("="*70)
print(f"\nTotal samples converted: {total_samples}")
print(f"\nReady-to-use datasets:")
print(f"  Training: {train_path}")
print(f"  Validation: {val_path}")
print(f"\nThese files are now in proper Prometheus format for training!")
print("="*70)
