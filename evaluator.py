# Copyright (c) 2025 VortexBench Team
# SPDX-License-Identifier: Apache-2.0

import logging
from prompts import (
    prompt_interleaved_reasoning_physical,
    prompt_interleaved_reasoning_embodied,
    prompt_interleaved_reasoning_logical,
    prompt_interleaved_reasoning_jigsaw,
    prompt_interleaved_reasoning_multi_view,
    prompt_reasoning_alignment_physical,
    prompt_reasoning_alignment_logical,
    prompt_reasoning_alignment_jigsaw,
    prompt_reasoning_alignment_multi_view,
)
from base_metric import (
    METRICS,
    get_task_data,
    get_image_paths,
    load_think_output,
    validate_inputs,
    encode_image_to_base64,
    evaluate_interleaved_reasoning,
    evaluate_reasoning_alignment,
)
from config import MAX_RETRIES

# Problem type to prompt mapping
PROBLEM_TYPE_PROMPTS = {
    "physical": {
        "interleaved_reasoning": prompt_interleaved_reasoning_physical,
        "reasoning_alignment": prompt_reasoning_alignment_physical,
    },
    "embodied": {
        "interleaved_reasoning": prompt_interleaved_reasoning_embodied,
        "reasoning_alignment": prompt_reasoning_alignment_physical,  # Same alignment prompt as physical
    },
    "logical": {
        "interleaved_reasoning": prompt_interleaved_reasoning_logical,
        "reasoning_alignment": prompt_reasoning_alignment_logical,
    },
    "jigsaw": {
        "interleaved_reasoning": prompt_interleaved_reasoning_jigsaw,
        "reasoning_alignment": prompt_reasoning_alignment_jigsaw,
    },
    "multi-view": {
        "interleaved_reasoning": prompt_interleaved_reasoning_multi_view,
        "reasoning_alignment": prompt_reasoning_alignment_multi_view,
    },
}

def evaluate_images(image_id, metrics=None, rover_data=None, api_key=None):
    """
    Unified evaluation function for ROVER problems
    
    Args:
        image_id: ID of the task to evaluate
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
    
    # Get problem type
    problem_type = task.get("problem_type")
    if not problem_type:
        logging.error(f"No problem_type found for task {image_id}")
        return results
    
    # Get prompts for this problem type
    prompts = PROBLEM_TYPE_PROMPTS.get(problem_type)
    if not prompts:
        logging.error(f"No prompts found for problem_type: {problem_type}")
        return results
    
    # Get file paths
    generated_path, answer_text_path = get_image_paths(image_id)
    
    # Validate that generated image exists
    if not task:
        logging.warning(f"Task not found")
        return results
    
    import os
    if not os.path.isfile(generated_path):
        logging.error(f"Generated image not found: {generated_path}")
        return results

    # Load model's answer text from gen_{id}.txt
    model_answer = load_think_output(answer_text_path)
    if not model_answer:
        model_answer = "No answer text available"

    # Encode generated image
    gen_b64 = encode_image_to_base64(generated_path)
    if not gen_b64:
        logging.error(f"Failed to encode generated image")
        return results

    # Extract task information
    prompt = task.get("prompt", "")
    answer = task.get("answer", "")
    
    # Get original problem image from HF dataset (PIL Image object)
    original_image = task.get('image')
    original_image_b64 = None
    if original_image is not None:
        original_image_b64 = encode_image_to_base64(original_image)
    
    # Get reasoning_image from HF dataset (PIL Image object)
    reasoning_image = task.get('reasoning_image')
    
    # Encode reasoning image if available
    reasoning_image_b64 = None
    if reasoning_image is not None:
        reasoning_image_b64 = encode_image_to_base64(reasoning_image)
    
    # Get image2 from HF dataset (PIL Image object) - for perception_next
    image2 = task.get('image2')
    image2_b64 = None
    if image2 is not None:
        image2_b64 = encode_image_to_base64(image2)

    # Evaluate each metric
    for metric in metrics:
        try:
            if metric == "interleaved_reasoning":
                prompt_text = prompts["interleaved_reasoning"].format(
                    prompt=prompt,
                    answer=answer
                )
                
                # Different problem types need different images
                if problem_type == "logical":
                    # Logical: problem image, GT reasoning, model reasoning
                    if reasoning_image_b64 is None:
                        logging.warning(f"No reasoning_image for logical task {image_id}, skipping interleaved_reasoning")
                        results["interleaved_reasoning_score"] = None
                        results["interleaved_reasoning_reasoning"] = "No reasoning image available"
                        continue
                    
                    score, reason = evaluate_interleaved_reasoning(
                        prompt_text, 
                        original_image_b64,  # Image 1: problem image
                        reasoning_image_b64,  # Image 2: GT reasoning  
                        gen_b64,  # Image 3: model reasoning
                        max_retries=MAX_RETRIES
                    )
                    
                elif problem_type == "jigsaw":
                    # Jigsaw puzzle: GT complete image, model's completion
                    if reasoning_image_b64 is None:
                        logging.warning(f"No reasoning_image for jigsaw task {image_id}, skipping interleaved_reasoning")
                        results["interleaved_reasoning_score"] = None
                        results["interleaved_reasoning_reasoning"] = "No reasoning image available"
                        continue
                    score, reason = evaluate_interleaved_reasoning(
                        prompt_text, 
                        reasoning_image_b64,  # Image 1: GT complete image
                        gen_b64,  # Image 2: model's generated completion
                        None,  # No third image
                        max_retries=MAX_RETRIES
                    )
                    
                elif problem_type == "multi-view":
                    # Multi-view: two views + model's generated wider view
                    score, reason = evaluate_interleaved_reasoning(
                        prompt_text, 
                        original_image_b64,  # Image 1: view 1
                        image2_b64,  # Image 2: view 2
                        gen_b64,  # Image 3: model's generated wider view
                        max_retries=MAX_RETRIES
                    )
                    
                elif problem_type == "physical":
                    # Physical: original image, GT result, model result
                    if reasoning_image_b64 is None:
                        logging.warning(f"No reasoning_image for task {image_id}, skipping interleaved_reasoning")
                        results["interleaved_reasoning_score"] = None
                        results["interleaved_reasoning_reasoning"] = "No reasoning image available"
                        continue
                    
                    score, reason = evaluate_interleaved_reasoning(
                        prompt_text, 
                        original_image_b64,  # Image 1: original image (initial state)
                        reasoning_image_b64,  # Image 2: GT result (expected state)
                        gen_b64,  # Image 3: model's generated result
                        max_retries=MAX_RETRIES
                    )
                    
                elif problem_type == "embodied":
                    # Embodied: GT trajectory, model trajectory (no need for original)
                    if reasoning_image_b64 is None:
                        logging.warning(f"No reasoning_image for task {image_id}, skipping interleaved_reasoning")
                        results["interleaved_reasoning_score"] = None
                        results["interleaved_reasoning_reasoning"] = "No reasoning image available"
                        continue
                    
                    score, reason = evaluate_interleaved_reasoning(
                        prompt_text, 
                        reasoning_image_b64,  # Image 1: GT trajectory
                        gen_b64,  # Image 2: model trajectory
                        None,  # No third image
                        max_retries=MAX_RETRIES
                    )
                    
                else:
                    logging.error(f"Unknown problem_type: {problem_type}")
                    continue
                results["interleaved_reasoning_score"] = score
                results["interleaved_reasoning_reasoning"] = reason
                
            elif metric == "reasoning_alignment":
                # All problem types need answer parameter to check correctness
                prompt_text = prompts["reasoning_alignment"].format(
                    prompt=prompt,
                    answer=answer,
                    model_answer=model_answer
                )
                
                # All problem types need original image + generated image for context
                if problem_type == "jigsaw":
                    # Jigsaw: original with gray box + generated completion
                    score, reason = evaluate_reasoning_alignment(
                        prompt_text, 
                        original_image_b64,  # Image 1: original with gray box (shows problem)
                        gen_b64,  # Image 2: model's completion
                        None,  # No third image
                        max_retries=MAX_RETRIES
                    )
                elif problem_type == "multi-view":
                    # Multi-view: two camera views + generated wider view
                    score, reason = evaluate_reasoning_alignment(
                        prompt_text, 
                        original_image_b64,  # Image 1: view 1 (shows problem)
                        image2_b64,  # Image 2: view 2 (shows problem)
                        gen_b64,  # Image 3: model's wider view
                        max_retries=MAX_RETRIES
                    )
                elif problem_type == "logical":
                    # Logical: original image + GT reasoning + model's reasoning
                    score, reason = evaluate_reasoning_alignment(
                        prompt_text, 
                        original_image_b64,  # Image 1: original problem image
                        reasoning_image_b64,  # Image 2: GT reasoning image (for comparison)
                        gen_b64,  # Image 3: model's generated reasoning image
                        max_retries=MAX_RETRIES
                    )
                else:
                    # Physical/Embodied: original image + generated image
                    score, reason = evaluate_reasoning_alignment(
                        prompt_text, 
                        original_image_b64,  # Image 1: original problem image
                        gen_b64,  # Image 2: model's generated image
                        None,  # No third image
                        max_retries=MAX_RETRIES
                    )
                results["reasoning_alignment_score"] = score
                results["reasoning_alignment_reasoning"] = reason

        except Exception as e:
            logging.error(f"Error evaluating {metric} for {image_id}: {e}")
            # Set default values for failed metrics (None to exclude from average)
            if metric == "interleaved_reasoning":
                results["interleaved_reasoning_score"] = None
                results["interleaved_reasoning_reasoning"] = f"Error: {str(e)}"
            elif metric == "reasoning_alignment":
                results["reasoning_alignment_score"] = None
                results["reasoning_alignment_reasoning"] = f"Error: {str(e)}"

    return results
