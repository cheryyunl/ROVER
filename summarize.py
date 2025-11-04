# Copyright (c) 2025 ROVER Team
# SPDX-License-Identifier: Apache-2.0

import os
import json
import argparse
from collections import defaultdict

# ROVER structure definitions
ROVER_DIMENSIONS = ["science", "culture", "common_sense", "logic"]
ROVER_REASONING_TYPES = ["temporal", "spatial", "quantitative", "causal", "synthetic", "logical", "mathematical", "abstract"] 
ROVER_METRICS = ["reasoning_process", "reasoning_visual", "reasoning_alignment", "visual_consistency", "image_quality"]

# Score names for display
METRIC_DISPLAY_NAMES = {
    "reasoning_process": "RP",
    "reasoning_visual": "RV", 
    "reasoning_alignment": "RA",
    "visual_consistency": "VC",
    "image_quality": "IQ",
    "average": "AVG"
}

def parse_task_id(task_id):
    """Parse task ID to extract dimension and reasoning_type"""
    # Expected formats:
    # 1. {dimension}_{reasoning_type}_{number} - e.g., "science_temporal_1" -> ("science", "temporal")
    # 2. {dimension}_{reasoning_type}_{number} - e.g., "logic_abstract_1" -> ("logic", "abstract")
    # 3. {dimension}_{reasoning_type}_{number} - e.g., "logic_math_1" -> ("logic", "mathematical")
    # 4. {dimension}_{reasoning_type}_{number} - e.g., "common_sense_causal_1" -> ("common_sense", "causal")
    
    parts = task_id.split('_')
    if len(parts) >= 3:
        # Handle special case: "common_sense" dimension
        if parts[0] == "common" and parts[1] == "sense":
            dimension = "common_sense"
            reasoning_type = parts[2]
        else:
            dimension = parts[0]
            reasoning_type = parts[1]
        
        # Handle special case: "math" -> "mathematical"
        if reasoning_type == "math":
            reasoning_type = "mathematical"
            
        return dimension, reasoning_type
    elif len(parts) >= 2:
        dimension = parts[0]
        reasoning_type = parts[1]
        
        # Handle special case: "math" -> "mathematical"
        if reasoning_type == "math":
            reasoning_type = "mathematical"
            
        return dimension, reasoning_type
    return None, None

def normalize_score(score):
    """Normalize a score from 1-5 scale to 0-100 scale"""
    if score is None:
        return None
    return (score - 1) * 25

def load_rover_results(jsonl_path):
    """Load and parse ROVER evaluation results"""
    results = {}
    
    if not os.path.exists(jsonl_path):
        print(f"Results file not found: {jsonl_path}")
        return results
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                task_id = data["key"]
                result = data["result"]
                
                # Parse task information
                dimension, reasoning_type = parse_task_id(task_id)
                if not dimension or not reasoning_type:
                    print(f"Warning: Could not parse task ID: {task_id}")
                    continue
                
                # Store normalized scores
                task_result = {
                    "task_id": task_id,
                    "dimension": dimension,
                    "reasoning_type": reasoning_type
                }
                
                for metric in ROVER_METRICS:
                    # Try both formats: "metric_score" and "metric": {"score": value}
                    score_key = f"{metric}_score"
                    if score_key in result and result[score_key] is not None:
                        # Format: "reasoning_visual_score": 3.0
                        normalized_score = normalize_score(result[score_key])
                        task_result[metric] = normalized_score
                    elif metric in result and isinstance(result[metric], dict) and "score" in result[metric]:
                        # Format: "reasoning_visual": {"score": 3.0, "explanation": "..."}
                        score_value = result[metric]["score"]
                        if score_value is not None:
                            # Check if score is already normalized (0-100) or needs normalization (1-5)
                            if score_value > 5:
                                # Already normalized to 0-100, use as is
                                task_result[metric] = score_value
                            else:
                                # Normalize from 1-5 to 0-100
                                normalized_score = normalize_score(score_value)
                                task_result[metric] = normalized_score
                        else:
                            task_result[metric] = None
                    else:
                        task_result[metric] = None
                
                results[task_id] = task_result
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Error parsing line: {e}")
                continue
    
    return results

def calculate_summary_statistics(results):
    """Calculate summary statistics for ROVER results"""
    
    # Group results by dimension and reasoning_type
    dimension_scores = defaultdict(lambda: defaultdict(list))  # dimension -> metric -> [scores]
    reasoning_type_scores = defaultdict(lambda: defaultdict(list))  # reasoning_type -> metric -> [scores]
    overall_scores = defaultdict(list)  # metric -> [scores]
    
    for task_id, result in results.items():
        dimension = result["dimension"]
        reasoning_type = result["reasoning_type"]
        
        for metric in ROVER_METRICS:
            score = result.get(metric)
            if score is not None:
                dimension_scores[dimension][metric].append(score)
                reasoning_type_scores[reasoning_type][metric].append(score)
                overall_scores[metric].append(score)
    
    # Calculate averages
    def calc_average(scores_dict):
        averages = {}
        all_scores = []
        
        for metric in ROVER_METRICS:
            scores = scores_dict.get(metric, [])
            if scores:
                avg = sum(scores) / len(scores)
                averages[METRIC_DISPLAY_NAMES[metric]] = avg
                all_scores.extend(scores)
        
        if all_scores:
            averages[METRIC_DISPLAY_NAMES["average"]] = sum(all_scores) / len(all_scores)
        
        return averages
    
    # Calculate dimension averages
    dimension_summary = {}
    for dimension in ROVER_DIMENSIONS:
        if dimension in dimension_scores:
            dimension_summary[dimension] = calc_average(dimension_scores[dimension])
    
    # Calculate reasoning_type averages
    reasoning_type_summary = {}
    for reasoning_type in ROVER_REASONING_TYPES:
        if reasoning_type in reasoning_type_scores:
            reasoning_type_summary[reasoning_type] = calc_average(reasoning_type_scores[reasoning_type])
    
    # Calculate overall average
    overall_summary = calc_average(overall_scores)
    
    return {
        "dimensions": dimension_summary,
        "reasoning_types": reasoning_type_summary,
        "overall": overall_summary,
        "task_count": len(results)
    }

def print_summary_report(summary):
    """Print formatted summary report"""
    print("="*60)
    print("ROVER Evaluation Summary")
    print("="*60)
    print(f"Total tasks evaluated: {summary['task_count']}")
    print()
    
    # Print dimensions
    print("Results by Dimension:")
    print("-" * 40)
    for dimension in ROVER_DIMENSIONS:
        if dimension in summary["dimensions"]:
            scores = summary["dimensions"][dimension]
            print(f"{dimension.capitalize()}:")
            for metric_key in ["RP", "RV", "RA", "VC", "IQ", "AVG"]:
                if metric_key in scores:
                    print(f"  {metric_key}: {scores[metric_key]:.2f}")
            print()
    
    # Print reasoning types
    print("Results by Reasoning Type:")
    print("-" * 40)
    for reasoning_type in ROVER_REASONING_TYPES:
        if reasoning_type in summary["reasoning_types"]:
            scores = summary["reasoning_types"][reasoning_type]
            print(f"{reasoning_type.capitalize()}:")
            for metric_key in ["RP", "RV", "RA", "VC", "IQ", "AVG"]:
                if metric_key in scores:
                    print(f"  {metric_key}: {scores[metric_key]:.2f}")
            print()
    
    # Print overall
    print("Overall Results:")
    print("-" * 40)
    overall = summary["overall"]
    for metric_key in ["RP", "RV", "RA", "VC", "IQ", "AVG"]:
        if metric_key in overall:
            print(f"{metric_key}: {overall[metric_key]:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Summarize ROVER evaluation results")
    parser.add_argument('--results_file', type=str, required=True, 
                        help='Path to the rover_metrics.jsonl file')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory to save summary results')
    
    args = parser.parse_args()
    
    # Load results
    print("Loading ROVER results...")
    results = load_rover_results(args.results_file)
    
    if not results:
        print("No valid results found!")
        return
    
    print(f"Loaded {len(results)} task results")
    
    # Calculate summary statistics
    print("Calculating summary statistics...")
    summary = calculate_summary_statistics(results)
    
    # Print report
    print_summary_report(summary)
    
    # Save detailed results to JSON
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'rover_summary.json')
    summary_data = {
        "summary": summary,
        "detailed_results": results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {output_path}")

if __name__ == "__main__":
    main()