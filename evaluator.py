# Copyright (c) 2025 ROVER Team
# SPDX-License-Identifier: Apache-2.0

import logging
from prompts import (
    prompt_reasoning_process_temporal,
    prompt_reasoning_visual_temporal,
    prompt_reasoning_process_spatial,
    prompt_reasoning_visual_spatial,
    prompt_reasoning_process_quantitative,
    prompt_reasoning_visual_quantitative,
    prompt_reasoning_process_causal,
    prompt_reasoning_visual_causal,
    prompt_reasoning_process_synthetic,
    prompt_reasoning_visual_synthetic,
    prompt_reasoning_process_logical,
    prompt_reasoning_visual_logical,
    prompt_reasoning_alignment,
    prompt_visual_consistency,
    prompt_image_quality,
)
from base_metric import (
    METRICS,
    get_task_data,
    get_image_paths,
    load_think_output,
    validate_inputs,
    encode_images,
    evaluate_reasoning_process,
    evaluate_reasoning_visual,
    evaluate_reasoning_alignment,
    evaluate_visual_consistency,
    evaluate_image_quality,
)
from config import MAX_RETRIES

# Reasoning type to prompt mapping
REASONING_PROMPTS = {
    "temporal": {
        "process": prompt_reasoning_process_temporal,
        "visual": prompt_reasoning_visual_temporal,
    },
    "spatial": {
        "process": prompt_reasoning_process_spatial,
        "visual": prompt_reasoning_visual_spatial,
    },
    "quantitative": {
        "process": prompt_reasoning_process_quantitative,
        "visual": prompt_reasoning_visual_quantitative,
    },
    "causal": {
        "process": prompt_reasoning_process_causal,
        "visual": prompt_reasoning_visual_causal,
    },
    "synthetic": {
        "process": prompt_reasoning_process_synthetic,
        "visual": prompt_reasoning_visual_synthetic,
    },
    "logical": {
        "process": prompt_reasoning_process_logical,
        "visual": prompt_reasoning_visual_logical,
    },
    "mathematical": {  # Map mathematical to logical
        "process": prompt_reasoning_process_logical,
        "visual": prompt_reasoning_visual_logical,
    },
    "abstract": {  # Map abstract to logical
        "process": prompt_reasoning_process_logical,
        "visual": prompt_reasoning_visual_logical,
    },
}

def evaluate_images(image_id, metrics=None, rover_data=None, api_key=None):
    """
    Unified evaluation function for all reasoning types
    
    Args:
        image_id: ID of the image to evaluate
        metrics: List of metrics to evaluate (default: all metrics)
        rover_data: Dataset containing task information
        api_key: API key (deprecated, kept for compatibility)
    
    Returns:
        dict: Evaluation results with scores and reasoning for each metric
    """
    metrics = metrics or METRICS
    results = {}
    
    # Find the specific task
    task = get_task_data(rover_data, image_id)
    if not task:
        logging.warning(f"Task ID {image_id} not found")
        return results
    
    # Get reasoning type
    reasoning_type = task.get("reasoning_type")
    if not reasoning_type:
        logging.error(f"No reasoning_type found for task {image_id}")
        return results
    
    # Get prompts for this reasoning type
    prompts = REASONING_PROMPTS.get(reasoning_type)
    if not prompts:
        logging.error(f"No prompts found for reasoning_type: {reasoning_type}")
        return results
    
    # Get images from HF dataset (PIL Image objects) and file paths
    original_image = task.get('image')  # PIL Image object from HF dataset
    target_image = task.get('target_image')  # PIL Image object from HF dataset
    
    generated_path, think_path = get_image_paths(image_id)

    # Validate inputs
    if not validate_inputs(task, generated_path, original_image):
        return results
    
    # Load think output if exists
    think_output = load_think_output(think_path)

    # Encode images
    orig_b64, gen_b64, target_b64 = encode_images(original_image, generated_path, target_image)
    if orig_b64 is None:
        return results

    # Extract task information
    prompt = task.get("prompt", "")
    dimension = task.get("dimension", "")
    keywords = task.get("keywords", "")  # Already a string in HF dataset
    target_description = task.get("target_description", "")
    
    # Keywords is already a string, use directly
    keywords_str = keywords if keywords else ""

    # Evaluate each metric
    for metric in metrics:
        try:
            if metric == "reasoning_process":
                prompt_text = prompts["process"].format(
                    prompt=prompt,
                    dimension=dimension,
                    keywords=keywords_str,
                    target_description=target_description,
                    think_output=think_output if think_output else "No think output available"
                )
                score, reason = evaluate_reasoning_process(prompt_text, orig_b64, max_retries=MAX_RETRIES)
                results["reasoning_process_score"] = score
                results["reasoning_process_reasoning"] = reason
                
            elif metric == "reasoning_visual":
                prompt_text = prompts["visual"].format(
                    prompt=prompt,
                    dimension=dimension,
                    keywords=keywords_str,
                    target_description=target_description
                )
                score, reason = evaluate_reasoning_visual(prompt_text, orig_b64, gen_b64, target_b64, max_retries=MAX_RETRIES)
                results["reasoning_visual_score"] = score
                results["reasoning_visual_reasoning"] = reason

            elif metric == "reasoning_alignment":
                prompt_text = prompt_reasoning_alignment.format(
                    prompt=prompt,
                    think_output=think_output if think_output else "No think output available"
                )
                score, reason = evaluate_reasoning_alignment(prompt_text, orig_b64, gen_b64, max_retries=MAX_RETRIES)
                results["reasoning_alignment_score"] = score
                results["reasoning_alignment_reasoning"] = reason

            elif metric == "visual_consistency":
                prompt_text = prompt_visual_consistency.format(prompt=prompt)
                score, reason = evaluate_visual_consistency(prompt_text, orig_b64, gen_b64, max_retries=MAX_RETRIES)
                results["visual_consistency_score"] = score
                results["visual_consistency_reasoning"] = reason

            elif metric == "image_quality":
                score, reason = evaluate_image_quality(prompt_image_quality, gen_b64, max_retries=MAX_RETRIES)
                results["image_quality_score"] = score
                results["image_quality_reasoning"] = reason

        except Exception as e:
            logging.error(f"Error evaluating {metric} for {image_id}: {e}")
            # Set default values for failed metrics
            if metric == "reasoning_process":
                results["reasoning_process_score"] = 0
                results["reasoning_process_reasoning"] = f"Error: {str(e)}"
            elif metric == "reasoning_visual":
                results["reasoning_visual_score"] = 0
                results["reasoning_visual_reasoning"] = f"Error: {str(e)}"
            elif metric == "reasoning_alignment":
                results["reasoning_alignment_score"] = 0
                results["reasoning_alignment_reasoning"] = f"Error: {str(e)}"
            elif metric == "visual_consistency":
                results["visual_consistency_score"] = 0
                results["visual_consistency_reasoning"] = f"Error: {str(e)}"
            elif metric == "image_quality":
                results["image_quality_score"] = 0
                results["image_quality_reasoning"] = f"Error: {str(e)}"

    return results
