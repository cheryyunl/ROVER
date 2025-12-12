import os
import re
import argparse
from openai import OpenAI
from PIL import Image
from io import BytesIO
from datasets import load_dataset
import time
import base64

# Initialize client
client = OpenAI()

# System prompt
SYSTEM_PROMPT = '''You are an image generation model. You should first think about the image content and prompt in the mind, and then you must generate the image for the prompt. 
Enclose your thinking process within <think> </think> tags, then create a new image.
Format: <think> your analysis and thinking process here </think> [generated image]'''

def extract_thinking_text(text):
    """Extract content within <think> tags from text"""
    pattern = r'<think>(.*?)</think>'
    matches = re.findall(pattern, text, re.DOTALL)
    return '\n'.join(matches).strip() if matches else ""

def file_exists_and_valid(png_path, txt_path):
    """Check if file exists and is valid"""
    return os.path.exists(png_path) and os.path.getsize(png_path) > 0

def pil_to_base64(image):
    """Convert PIL image to base64 string"""
    byte_arr = BytesIO()
    image.save(byte_arr, format='PNG')
    byte_arr.seek(0)
    return base64.b64encode(byte_arr.getvalue()).decode('utf-8')

def process_benchmark(max_samples=None, output_dir="./gen_gpt", task_id=None):
    """Process benchmark dataset
    
    Args:
        max_samples: Maximum number of samples to process, None means process all
        output_dir: Output directory path
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dataset
        print("Loading dataset cheryyunl/ROVER...")
        dataset = load_dataset("cheryyunl/ROVER", "ROVER-IG")
        
        # Debug: View dataset structure
        print("Dataset keys:", list(dataset.keys()))
        
        # Assume data is in the default split; if not, adjustment might be needed
        data_split = dataset['train'] if 'train' in dataset else dataset[list(dataset.keys())[0]]
        
        print(f"Dataset loaded, total {len(data_split)} samples")
        
        # Show basic dataset information
        if len(data_split) > 0:
            first_item = data_split[0]
            print("Sample fields:", list(first_item.keys()))
        
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        # HuggingFace Dataset needs select instead of slicing
        if task_id is not None:
            print(f"Processing only task ID: {task_id}")
            # Find the specified task
            items_to_process = [item for item in data_split if item['id'] == task_id]
            if not items_to_process:
                print(f"Error: Task ID '{task_id}' not found")
                return
            total_to_process = 1
        elif max_samples is not None:
            total_to_process = min(max_samples, len(data_split))
            print(f"Limiting to first {max_samples} samples, actually processing {total_to_process}")
        else:
            total_to_process = len(data_split)
            print(f"Processing all {total_to_process} samples")
            items_to_process = data_split
        
        for idx, item in enumerate(items_to_process):
            try:
                # Get basic information
                task_id = item['id']
                image = item['image']  # Should already be PIL Image
                prompt = item['prompt']
                
                # Generate output file paths
                png_path = os.path.join(output_dir, f"gen_{task_id}.png")
                txt_path = os.path.join(output_dir, f"gen_{task_id}.txt")
                
                # Check if valid output files already exist
                if file_exists_and_valid(png_path, txt_path):
                    print(f"[{idx+1}/{total_to_process}] Skipping {task_id}: Output file already exists")
                    skipped_count += 1
                    continue
                
                print(f"[{idx+1}/{total_to_process}] Processing {task_id}...")
                
                # Build complete prompt
                full_prompt = f"{SYSTEM_PROMPT}\n\nTask: {prompt}"
                
                # Attempt to generate content (with retry mechanism)
                max_retries = 2
                generated_image = None
                thinking_text = ""
                all_text = ""
                
                for attempt in range(max_retries + 1):
                    try:
                        # Convert image to base64
                        base64_image = pil_to_base64(image)
                        
                        response = client.responses.create(
                            model="gpt-5",
                            input=[
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "input_text",
                                            "text": full_prompt
                                        },
                                        {
                                            "type": "input_image",
                                            "image_url": f"data:image/png;base64,{base64_image}"
                                        }
                                    ]
                                }
                            ],
                            tools=[{"type": "image_generation"}]
                        )
                        
                        generated_image = None
                        thinking_text = ""
                        all_text = ""
                        
                        # Process all parts in the response
                        for output in response.output:
                            if hasattr(output, 'content'):
                                if isinstance(output.content, str):
                                    all_text += output.content
                                elif isinstance(output.content, list):
                                    for item in output.content:
                                        if isinstance(item, str):
                                            all_text += item
                                        elif hasattr(item, 'text'):
                                            all_text += item.text
                            
                            if hasattr(output, 'type') and output.type == "image_generation_call":
                                if hasattr(output, 'result') and output.result:
                                    image_bytes = base64.b64decode(output.result)
                                    generated_image = Image.open(BytesIO(image_bytes))
                        
                        # If image generated, break retry loop
                        if generated_image is not None:
                            break
                        elif attempt < max_retries:
                            print(f"  Attempt {attempt+1} did not generate an image, retrying...")
                            # Modify prompt to emphasize image generation
                            full_prompt = f"{SYSTEM_PROMPT}\n\nIMPORTANT: You MUST generate an image. Do not provide only text.\n\nTask: {prompt}"
                            time.sleep(2)
                        
                    except Exception as e:
                        print(f"  Error generating content (attempt {attempt+1}): {e}")
                        if attempt < max_retries:
                            time.sleep(2)
                        else:
                            raise e
                
                # Extract thinking process text
                thinking_text = extract_thinking_text(all_text)
                
                # If still no image generated, save all text for debugging
                if generated_image is None:
                    print(f"  Warning: No image generated after {max_retries+1} attempts")
                    print(f"  Returned text: {all_text[:200]}...")
                    # Save complete response text for debugging
                    thinking_text = f"[No image generated]\nOriginal response:\n{all_text}\n\nExtracted thinking process:\n{thinking_text}"
                
                # Save generated image
                if generated_image is not None:
                    generated_image.save(png_path)
                    print(f"  ✓ Image saved: {png_path}")
                    
                    # Save thinking process text
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(thinking_text)
                    print(f"  ✓ Text saved: {txt_path}")
                    
                    processed_count += 1
                else:
                    print(f"  Skipping task {task_id}: No image generated")
                    error_count += 1
                
                # Add small delay to avoid API rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"  Error processing sample {idx}: {e}")
                error_count += 1
                # Print detailed error information for debugging
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\nProcessing complete!")
        print(f"Successfully processed: {processed_count}")
        print(f"Skipped: {skipped_count}")
        print(f"Errors: {error_count}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ROVER-IG Benchmark Evaluation Script')
    parser.add_argument('--max-samples', type=int, default=10, 
                       help='Maximum number of samples to process (default: 10, set to 0 or negative for all)')
    parser.add_argument('--output-dir', type=str, default='./gen_gpt',
                       help='Output directory (default: ./gen_gpt)')
    parser.add_argument('--task-id', type=str, default=None,
                       help='Specify task ID to process (e.g., logic_abstract_1)')
    
    args = parser.parse_args()
    
    # Set processing quantity
    max_samples = args.max_samples if args.max_samples > 0 else None
    
    print("ROVER-IG Benchmark Evaluation Script")
    print("=" * 50)
    print(f"Output directory: {args.output_dir}")
    
    if max_samples:
        print(f"Limiting processing count: {max_samples}")
    else:
        print("Processing all samples")
    
    process_benchmark(max_samples=max_samples, output_dir=args.output_dir, task_id=args.task_id)
