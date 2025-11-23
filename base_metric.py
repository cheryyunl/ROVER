# Copyright (c) 2025 VortexBench Team
# SPDX-License-Identifier: Apache-2.0

import os
import json
import base64
import time
import re
import logging
from PIL import Image
from io import BytesIO
from openai import AzureOpenAI
import threading

# Import configuration
from config import OPENAI_API_KEY, OPENAI_MODEL, AZURE_API_KEY, AZURE_ENDPOINT, AZURE_DEPLOYMENT_NAME, AZURE_API_VERSION, VORTEX_GEN_DIR, MAX_TOKENS

# Thread-safe file writing lock
lock = threading.Lock()

# Initialize OpenAI client (prefer OpenAI over Azure)
if OPENAI_API_KEY:
    # Use standard OpenAI API
    from openai import OpenAI
    openai_client = OpenAI(
        api_key=OPENAI_API_KEY,
    )
    model_name = OPENAI_MODEL
else:
    # Fallback to Azure OpenAI
    openai_client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
    )
    model_name = AZURE_DEPLOYMENT_NAME

# Define metrics for ROVER evaluation
METRICS = ["interleaved_reasoning", "reasoning_alignment"]

def save_result_jsonl(result, key, output_jsonl_path):
    """Save evaluation result to JSONL file with thread lock"""
    with lock:
        with open(output_jsonl_path, 'a', encoding='utf-8') as f:
            data = {"key": key, "result": result}
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

def load_processed_keys_with_missing_metrics(jsonl_path, metrics, expected_keys_map):
    """Load processed image IDs and return missing metrics for each key"""
    key_missing_metrics = {}  # key -> list of missing metrics
    fully_completed_keys = set()  # keys that have all metrics completed
    
    if os.path.exists(jsonl_path):
        # First, collect all results for each key
        key_results = {}  # key -> merged result dict
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    key = data["key"]
                    result = data["result"]
                    
                    if key not in key_results:
                        key_results[key] = {}
                    key_results[key].update(result)
                except (json.JSONDecodeError, KeyError):
                    continue
        
        # Check which metrics are missing for each key
        for key in expected_keys_map:
            if key in key_results:
                missing_metrics = []
                for metric in metrics:
                    # Check if this metric has valid scores
                    if metric == "interleaved_reasoning":
                        if f"interleaved_reasoning_score" not in key_results[key] or key_results[key][f"interleaved_reasoning_score"] is None:
                            missing_metrics.append(metric)
                    elif metric == "reasoning_alignment":
                        if f"reasoning_alignment_score" not in key_results[key] or key_results[key][f"reasoning_alignment_score"] is None:
                            missing_metrics.append(metric)
                
                if missing_metrics:
                    key_missing_metrics[key] = missing_metrics
                else:
                    fully_completed_keys.add(key)
            else:
                key_missing_metrics[key] = metrics[:]  # All metrics missing
    else:
        # File doesn't exist, all keys need all metrics
        key_missing_metrics = {key: metrics[:] for key in expected_keys_map}
    
    return key_missing_metrics, fully_completed_keys

def encode_image_to_base64(image_input):
    """Encode image to base64 string
    
    Args:
        image_input: Can be a file path (string) or PIL Image object
    """
    try:
        if isinstance(image_input, str):
            # It's a file path
            with open(image_input, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        else:
            # It's a PIL Image object
            buffer = BytesIO()
            # Convert to RGB if necessary (handles RGBA, L, P, etc.)
            if image_input.mode != 'RGB':
                image_input = image_input.convert('RGB')
            image_input.save(buffer, format='JPEG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        logging.error(f"Error encoding image {image_input}: {e}")
        return None

def evaluate_with_gpt(prompt, original_base64=None, edited_base64=None, target_base64=None, max_retries=5):
    """Evaluate images using OpenAI GPT-4o with intelligent retry mechanism"""
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    
    if original_base64:
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{original_base64}"}
        })
    
    if edited_base64:
        messages[0]["content"].append({
            "type": "image_url", 
            "image_url": {"url": f"data:image/png;base64,{edited_base64}"}
        })
    
    if target_base64:
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{target_base64}"}
        })

    for attempt in range(max_retries):
        try:
            # o3 models use max_completion_tokens instead of max_tokens
            # Check if model is o3/o3-mini
            if 'o3' in model_name.lower():
                response = openai_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_completion_tokens=MAX_TOKENS,
                )
            else:
                response = openai_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=MAX_TOKENS,
                    temperature=0.0,
                )
            
            content = response.choices[0].message.content
            if content and content.strip():
                content = content.strip()
                time.sleep(1)  # Rate limiting
                return content  # Return content for extract_score_and_reason to handle
            else:
                logging.warning(f"Attempt {attempt + 1}: Empty response, retrying...")
                
            time.sleep(2 ** min(attempt, 3))  # Exponential backoff, capped at 8 seconds
            
        except Exception as e:
            logging.warning(f"OpenAI GPT evaluation attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** min(attempt, 3))  # Exponential backoff

    logging.error(f"OpenAI GPT evaluation failed after {max_retries} attempts.")
    return ""

def extract_score_and_reason(response, score_key, reason_fields, prefix_patterns=None):
    """Extract score and reasoning from GPT response with robust parsing"""
    if not response or not response.strip():
        return None, None
    
    # Clean up response
    response = response.strip()
    
    # Debug: Log the response for troubleshooting
    logging.debug(f"Extracting from response: {response[:200]}...")
    
    # Try multiple JSON parsing strategies
    score, reason = None, None
    
    # Strategy 1: Direct JSON parsing
    try:
        data = json.loads(response)
        score = data.get(score_key)
        for reason_field in reason_fields:
            reason = data.get(reason_field)
            if reason:
                break
        if score is not None:
            try:
                score_int = int(score)
                if 1 <= score_int <= 5:
                    logging.debug(f"Strategy 1 success: score={score_int}, reason={reason}")
                    return score_int, reason
                else:
                    logging.debug(f"Strategy 1: score {score_int} out of range [1-5]")
            except (ValueError, TypeError):
                logging.debug(f"Strategy 1: invalid score format '{score}'")
        else:
            logging.debug(f"Strategy 1: score_key '{score_key}' not found in {list(data.keys())}")
    except Exception as e:
        logging.debug(f"Strategy 1 failed: {e}")
        pass
    
    # Strategy 2: Handle double braces
    try:
        if response.startswith('{{') and response.endswith('}}'):
            clean_response = response[1:-1]
            data = json.loads(clean_response)
            score = data.get(score_key)
            for reason_field in reason_fields:
                reason = data.get(reason_field)
                if reason:
                    break
            if score is not None:
                return int(score), reason
    except:
        pass
    
    # Strategy 3: Extract JSON from text
    try:
        # Look for JSON-like patterns in the text
        json_pattern = r'\{[^{}]*"' + re.escape(score_key) + r'"[^{}]*\}'
        json_match = re.search(json_pattern, response, re.IGNORECASE)
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)
            score = data.get(score_key)
            for reason_field in reason_fields:
                reason = data.get(reason_field)
                if reason:
                    break
            if score is not None:
                return int(score), reason
    except:
        pass
    
    # Strategy 4: Regex fallback patterns
    # Convert score_key from "reasoning_process_score" to "reasoning process score"
    score_key_readable = score_key.replace('_', ' ')
    
    patterns = [
        rf"{score_key}\s*[:：]?\s*([1-5])",
        rf"{score_key_readable}\s*[:：]?\s*([1-5])",  # "reasoning process score: 4"
        rf"{score_key_readable}\s+is\s+([1-5])",      # "reasoning process score is 4"
        rf"{score_key_readable}\s+=\s*([1-5])",       # "reasoning process score = 4"
        r"([1-5])\s*/\s*5",
        r"([1-5])\s+out\s+of\s+5",
        r"score\s*[:：]?\s*([1-5])",
        r"rating\s*[:：]?\s*([1-5])",
        r"([1-5])\s+points?",           # "4 points"
        r"([1-5])\s+stars?",            # "4 stars"
        r"([1-5])\s*$",                 # "4" at end of line
        r"^([1-5])\s*",                 # "4" at start of line
        r"([1-5])\s*\.",                # "4." 
        r"([1-5])\s*,",                 # "4,"
        r"([1-5])\s*\)",                # "4)"
        r"([1-5])\s*\]",                # "4]"
        r"([1-5])\s*}",                 # "4}"
    ]
    if prefix_patterns:
        patterns = prefix_patterns + patterns
    
    for i, pat in enumerate(patterns):
        m = re.search(pat, response, re.IGNORECASE)
        if m:
            try:
                score = int(m.group(1))
                if 1 <= score <= 5:
                    logging.debug(f"Strategy 4 pattern {i+1} success: score={score}")
                    break
                else:
                    logging.debug(f"Strategy 4 pattern {i+1}: score {score} out of range [1-5]")
            except (ValueError, TypeError):
                logging.debug(f"Strategy 4 pattern {i+1}: invalid score format '{m.group(1)}'")
        else:
            logging.debug(f"Strategy 4 pattern {i+1} failed: {pat}")
    
    if score is None:
        logging.warning(f"All 4 strategies failed to extract score from response: {response[:500]}...")
        logging.warning(f"Looking for score_key: '{score_key}' in response")
    
    # Extract reasoning with multiple patterns
    reason_patterns = [
        r"reasoning\s*[:：]\s*(.+)",
        r"explanation\s*[:：]\s*(.+)",
        r"analysis\s*[:：]\s*(.+)",
        r"evaluation\s*[:：]\s*(.+)",
    ]
    
    for pat in reason_patterns:
        reason_match = re.search(pat, response, re.IGNORECASE|re.DOTALL)
        if reason_match:
            reason = reason_match.group(1).strip()
            # Clean up reasoning text
            if reason.startswith('"') and reason.endswith('"'):
                reason = reason[1:-1]
            break
    
    return score, reason

def get_task_data(vortex_data, image_id):
    """Get task data for a specific image ID"""
    for task in vortex_data["tasks"]:
        if task["id"] == image_id:
            return task
    return None

def get_image_paths(image_id):
    """Get file paths for generated image and think output"""
    generated_path = os.path.join(VORTEX_GEN_DIR, f"gen_{image_id}.png")
    think_path = os.path.join(VORTEX_GEN_DIR, f"gen_{image_id}.txt")
    return generated_path, think_path

def load_think_output(think_path):
    """Load think output from file if it exists"""
    if os.path.isfile(think_path):
        try:
            with open(think_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            logging.warning(f"Error loading think output {think_path}: {e}")
    return None

def validate_inputs(task, generated_path, original_image):
    """Validate that required inputs exist"""
    if not task:
        logging.warning(f"Task ID not found")
        return False
    
    if original_image is None:
        logging.error(f"Original image not found for task")
        return False
        
    if not os.path.isfile(generated_path):
        logging.error(f"Generated image not found: {generated_path}")
        return False
    
    return True

def encode_images(original_image, generated_path, target_image=None):
    """Encode all required images to base64"""
    orig_b64 = encode_image_to_base64(original_image)  # PIL Image object
    gen_b64 = encode_image_to_base64(generated_path)   # File path
    
    if not orig_b64 or not gen_b64:
        logging.error(f"Failed to encode images")
        return None, None, None
    
    # Encode target image if available
    target_b64 = None
    if target_image is not None:
        target_b64 = encode_image_to_base64(target_image)
    
    return orig_b64, gen_b64, target_b64

def evaluate_metric_with_retry(metric_name, prompt_text, orig_b64, gen_b64=None, target_b64=None, max_retries=3):
    """Evaluate a metric with retry logic for null scores"""
    score, reason = None, None
    for retry_attempt in range(max_retries):
        resp = evaluate_with_gpt(prompt_text, orig_b64, gen_b64, target_b64)
        if resp:  # Only try to parse if we got a response
            score, reason = extract_score_and_reason(resp, f"{metric_name}_score", ["reasoning"])
            if score is not None:  # Success, break out
                break
        logging.warning(f"{metric_name} evaluation attempt {retry_attempt + 1} failed to extract valid score, retrying...")
    
    return score, reason

def evaluate_interleaved_reasoning(prompt_text, image1_b64, image2_b64, image3_b64=None, max_retries=3):
    """Evaluate visual reasoning metric
    
    Args:
        prompt_text: Formatted evaluation prompt
        image1_b64: Base64 encoded first image (GT reasoning for physical/perception, problem image for logical)
        image2_b64: Base64 encoded second image (generated image for physical/perception, GT reasoning for logical)
        image3_b64: Base64 encoded third image (None for physical/perception, generated image for logical)
        max_retries: Number of retry attempts
    
    Returns:
        tuple: (score, reason)
    """
    return evaluate_metric_with_retry("interleaved_reasoning", prompt_text, image1_b64, image2_b64, image3_b64, max_retries=max_retries)

def evaluate_reasoning_alignment(prompt_text, image1_b64, image2_b64=None, image3_b64=None, max_retries=3):
    """Evaluate reasoning alignment metric
    
    Evaluates alignment between the problem prompt, generated image, and model's answer text.
    
    Args:
        prompt_text: Formatted evaluation prompt (includes problem prompt and model answer)
        image1_b64: Base64 encoded first image (generated for physical/embodied/logical, original for perception)
        image2_b64: Base64 encoded second image (optional)
        image3_b64: Base64 encoded third image (optional, for perception_next)
        max_retries: Number of retry attempts
    
    Returns:
        tuple: (score, reason)
    """
    return evaluate_metric_with_retry("reasoning_alignment", prompt_text, image1_b64, image2_b64, image3_b64, max_retries=max_retries)
