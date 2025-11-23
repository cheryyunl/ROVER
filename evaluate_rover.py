# Copyright (c) 2025 VortexBench Team
# SPDX-License-Identifier: Apache-2.0

import os
import json
import argparse
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datasets import load_dataset

# Import unified evaluator and config
from evaluator import evaluate_images
from config import VORTEX_GEN_DIR

# Hugging Face dataset
DATASET_NAME = "cheryyunl/ROVER"
SUBSET_NAME = "ROVER-TG"

METRICS = ["interleaved_reasoning", "reasoning_alignment"]


def save_result_jsonl(result, key, output_jsonl_path):
    """Save evaluation result to JSONL file"""
    with open(output_jsonl_path, 'a', encoding='utf-8') as f:
        data = {"key": key, "result": result}
        f.write(json.dumps(data, ensure_ascii=False) + '\n')


def process_task_evaluation(task, rover_data, metrics, api_key, output_jsonl_path):
    """Process single task evaluation"""
    try:
        task_id = task["id"]
        
        # Run evaluation with unified evaluator
        result = evaluate_images(
            image_id=task_id,
            metrics=metrics,
            rover_data=rover_data,
            api_key=api_key
        )
        
        # Save result
        save_result_jsonl(result, task_id, output_jsonl_path)
        return True
        
    except Exception as e:
        logging.error(f"Error processing task {task.get('id', 'unknown')}: {e}")
        return False


def load_huggingface_data():
    """Load data from Hugging Face dataset"""
    try:
        dataset = load_dataset(DATASET_NAME, SUBSET_NAME)
        print(f"Loaded dataset {DATASET_NAME} - {SUBSET_NAME}")
        return dataset
    except Exception as e:
        logging.error(f"Error loading Hugging Face dataset {DATASET_NAME}: {e}")
        return None

def convert_hf_to_rover_format(dataset):
    """Convert Hugging Face dataset to ROVER format"""
    tasks = []
    
    # Get the train split (or whatever split exists)
    split_data = dataset['train'] if 'train' in dataset else dataset
    
    for item in split_data:
        # Handle both 'reasoning_image' and 'refer_image' field names
        reasoning_image = item.get('reasoning_image') or item.get('refer_image')
        
        # Keep original problem_type for fine-grained prompt selection
        # Types: physical, embodied, logical, jigsaw, multi-view
        problem_type = item.get('problem_type', '')
        
        task = {
            'id': item.get('id'),
            'image': item.get('image'),  # PIL Image object (original problem image)
            'image2': item.get('image2'),  # PIL Image object (optional second image)
            'problem_type': problem_type,  # physical/embodied/logical/jigsaw/multi-view
            'prompt': item.get('prompt'),
            'reasoning_image': reasoning_image,  # PIL Image object (GT reasoning visualization)
            'answer': item.get('answer'),  # Ground truth answer (can be string or list)
        }
        tasks.append(task)
    
    return {'tasks': tasks}

def run_rover_evaluation(
    output_dir="rover_results",
    num_workers=10,
    metrics=None,
    api_key=None,
    filter_problem_type=None,
    force_reevaluate=False,
    max_tasks=None
):
    """
    Run ROVER evaluation using Hugging Face dataset
    
    Args:
        output_dir: Directory to save results
        num_workers: Number of parallel workers
        metrics: List of metrics to evaluate
        api_key: OpenAI API key
        filter_problem_type: Filter by problem type (physical/logical/perception)
        force_reevaluate: Force re-evaluation of already evaluated tasks
        max_tasks: Maximum number of tasks to evaluate (None for all)
    """
    metrics = metrics or METRICS
    
    # Setup output directory
    os.makedirs(output_dir, exist_ok=True)
    output_jsonl_path = os.path.join(output_dir, "rover_metrics.jsonl")
    
    # Load Hugging Face dataset
    dataset = load_huggingface_data()
    if dataset is None:
        return
    
    # Convert to ROVER format
    rover_data = convert_hf_to_rover_format(dataset)
    
    # Filter tasks
    tasks = rover_data["tasks"]
    if filter_problem_type:
        tasks = [t for t in tasks if t.get("problem_type") == filter_problem_type]
    
    print(f"Found {len(tasks)} tasks to evaluate")
    
    # Check which tasks have generated images and haven't been evaluated
    valid_tasks = []
    already_evaluated = set()
    
    # Load already evaluated tasks
    if os.path.exists(output_jsonl_path):
        with open(output_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    task_id = data.get('key')
                    if task_id:
                        already_evaluated.add(task_id)
                except json.JSONDecodeError:
                    continue
    
    for task in tasks:
        task_id = task["id"]
        gen_image_path = os.path.join(VORTEX_GEN_DIR, f"gen_{task_id}.png")
        
        if os.path.exists(gen_image_path):
            if task_id not in already_evaluated or force_reevaluate:
                valid_tasks.append(task)
            else:
                print(f"Skipping already evaluated task: {task_id}")
        else:
            print(f"Warning: Generated image not found for {task_id}")
    
    # Apply max_tasks limit if specified
    if max_tasks is not None and max_tasks > 0:
        original_count = len(valid_tasks)
        valid_tasks = valid_tasks[:max_tasks]
        print(f"Limited to {len(valid_tasks)} tasks (from {original_count} available)")
    
    print(f"Found {len(valid_tasks)} new tasks to evaluate")
    print(f"Skipped {len(already_evaluated)} already evaluated tasks")
    
    if not valid_tasks:
        print("No tasks with generated images found. Please run generation first.")
        return
    
    # Process with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        
        for task in valid_tasks:
            future = executor.submit(
                process_task_evaluation,
                task, rover_data, metrics, api_key, output_jsonl_path
            )
            futures.append(future)
        
        # Process results with progress bar
        successful = 0
        failed = 0
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating VortexBench"):
            try:
                success = future.result()
                if success:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                logging.error(f"Future failed: {e}")
                failed += 1
    
    print(f"Evaluation completed: {successful} successful, {failed} failed")
    print(f"Results saved to: {output_jsonl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ROVER Evaluation")
    parser.add_argument("--output_dir", type=str, default="rover_results", help="Output directory")
    parser.add_argument("--workers", type=int, default=10, help="Number of worker threads")
    parser.add_argument("--metrics", nargs="+", choices=METRICS, default=METRICS, help="Metrics to evaluate")
    parser.add_argument("--api_key", type=str, help="[DEPRECATED] API key parameter - credentials are configured in config.py")
    parser.add_argument("--problem_type", type=str, choices=["physical", "embodied", "logical", "jigsaw", "multi-view"], help="Filter by problem type")
    parser.add_argument("--force_reevaluate", action="store_true", help="Force re-evaluation of already evaluated tasks")
    parser.add_argument("--max_tasks", type=int, help="Maximum number of tasks to evaluate (useful for testing)")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # API key handling (deprecated - now configured in config.py)
    api_key = args.api_key
    if api_key:
        print("Warning: --api_key parameter is deprecated. Credentials are configured in config.py")
    
    run_rover_evaluation(
        output_dir=args.output_dir,
        num_workers=args.workers,
        metrics=args.metrics,
        api_key=api_key,
        filter_problem_type=args.problem_type,
        force_reevaluate=args.force_reevaluate,
        max_tasks=args.max_tasks
    )
