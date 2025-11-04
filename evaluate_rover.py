# Copyright (c) 2025 ROVER Team
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
from config import ROVER_GEN_DIR

# Hugging Face dataset
DATASET_NAME = "cheryyunl/ROVER-Gen"

METRICS = ["reasoning_process", "reasoning_visual", "reasoning_alignment", "visual_consistency", "image_quality"]


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
        dataset = load_dataset(DATASET_NAME)
        print(f"Loaded dataset {DATASET_NAME}")
        return dataset
    except Exception as e:
        logging.error(f"Error loading Hugging Face dataset {DATASET_NAME}: {e}")
        return None

def convert_hf_to_rover_format(dataset):
    """Convert Hugging Face dataset to ROVER format"""
    tasks = []
    
    # Get the train split
    split_data = dataset['train'] if 'train' in dataset else dataset
    
    # Get dimension labels mapping
    dimension_names = split_data.features['dimension'].names
    
    for item in split_data:
        # Extract and process fields
        keywords = item.get('keywords', '')  # This is already a string
        target_description = item.get('target_description', '')
        
        # Convert dimension index to name
        dimension_idx = item.get('dimension')
        dimension = dimension_names[dimension_idx] if dimension_idx is not None else 'unknown'
        
        task = {
            'id': item.get('id'),
            'image_file': item.get('image_file'),
            'dimension': dimension,  # Convert index to name
            'reasoning_type': item.get('reasoning_type'),
            'prompt': item.get('prompt'),
            'target_description': target_description,
            'keywords': keywords,  # Already a string
            'image': item.get('image'),  # PIL Image object
            'target_image': item.get('target_image'),  # PIL Image object (if exists)
        }
        tasks.append(task)
    
    return {'tasks': tasks}

def run_rover_evaluation(
    output_dir="rover_results",
    num_workers=10,
    metrics=None,
    api_key=None,
    filter_dimension=None,
    filter_reasoning_type=None,
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
        filter_dimension: Filter by dimension (science/humanity/common_sense/logic)
        filter_reasoning_type: Filter by reasoning type (temporal/spatial/quantitative/causal/synthetic)
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
    if filter_dimension:
        tasks = [t for t in tasks if t.get("dimension") == filter_dimension]
    if filter_reasoning_type:
        tasks = [t for t in tasks if t.get("reasoning_type") == filter_reasoning_type]
    
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
        gen_image_path = os.path.join(ROVER_GEN_DIR, f"gen_{task_id}.png")
        
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
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating ROVER"):
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
    parser.add_argument("--api_key", type=str, help="[DEPRECATED] API key parameter - Azure credentials are configured in metric files")
    parser.add_argument("--dimension", type=str, choices=["science", "humanity", "common_sense", "logic"], help="Filter by dimension")
    parser.add_argument("--reasoning_type", type=str, choices=["temporal", "spatial", "quantitative", "causal", "synthetic"], help="Filter by reasoning type")
    parser.add_argument("--force_reevaluate", action="store_true", help="Force re-evaluation of already evaluated tasks")
    parser.add_argument("--max_tasks", type=int, help="Maximum number of tasks to evaluate (useful for testing)")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # API key handling (deprecated - now configured in metric files)
    api_key = args.api_key
    if api_key:
        print("Warning: --api_key parameter is deprecated. Azure credentials are configured in metric files.")
    
    run_rover_evaluation(
        output_dir=args.output_dir,
        num_workers=args.workers,
        metrics=args.metrics,
        api_key=api_key,
        filter_dimension=args.dimension,
        filter_reasoning_type=args.reasoning_type,
        force_reevaluate=args.force_reevaluate,
        max_tasks=args.max_tasks
    )
