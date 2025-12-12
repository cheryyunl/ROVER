import os
import argparse
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
from datasets import load_dataset
import time

# Initialize client
client = genai.Client()

# Embodied task system prompt
EMBODIED_SYSTEM_PROMPT = '''You are a robotics trajectory planner with visualization capabilities. When given a robotics scene, you must:

1. Generate a trajectory visualization image:
   - Overlay 10 waypoint markers on the input scene
   - Style: Blue circles with white outlines + connecting trajectory lines
   - Labels: 'traj1', 'traj2', ..., 'traj10'
   - Reference the example image (Image 2) for exact visualization style

2. Output pixel coordinates based on your visualization:
   - Format: [[x1, y1], [x2, y2], ..., [x10, y10]]
   - Coordinate system: (0, 0) at top-left corner

Constraints:
- Start from current end-effector position
- End at task completion position
- Generate smooth, collision-free waypoints
- All coordinates must be within image boundaries

Format your output as:
<think>
[Your analysis and planning process]
</think>

Output:
(1) Generated trajectory visualization image
(2) Trajectory coordinates: [[x1, y1], [x2, y2], ..., [x10, y10]]'''

# Physical task system prompt
PHYSICAL_SYSTEM_PROMPT = '''You are a physics simulation AI with image generation capabilities. Given a physical scenario, you must:

**REQUIRED OUTPUT:**
1. Generate a simulation result image that visualizes what happens after the physics plays out
2. Provide reasoning for your prediction
3. Give the final answer to the question

**HOW TO APPROACH:**

Step 1 - Analyze the initial scene (Image 1):
- What objects are present? (cars, balls, liquids, blocks, pulleys, etc.)
- What are their initial positions and states?
- What forces or motions will be applied?

Step 2 - Use Image 2 (if provided):
- It may show additional context of the scene

Step 3 - Generate a physics simulation result image:
Simulate what happens when physics is applied and generate the resulting scene:
- Apply gravity, momentum, friction, collision dynamics
- Simulate the complete physical process (cars moving and colliding, balls falling, liquids flowing, pulleys rotating)
- Generate an image showing the final state or outcome after physics simulation
- The simulated image should naturally show where objects end up, what gets affected, and what the result is

**EXAMPLES:**
- Car collision → Simulate the car moving and show which cubes are displaced/moved after collision
- Ball falling → Simulate the ball's trajectory and show it in the final position (which pit)
- Liquid flow → Simulate the liquid flowing and show which containers have received the liquid
- Pulley system → Simulate the pulley rotation and show objects in their new positions

Format your output as:
<think>
1. Initial state: [describe the setup]
2. Generate the physics simulation image
3. Analyze the generated image: [what does the simulation show?]  
4. Determine answer: [based on the generated image, what is the result?]
</think>

Output:
(1) Generated physics simulation result image (showing the outcome after simulation)
(2) Image Analysis: [Carefully observe your generated simulated image - what is the final state? Where are objects positioned? What happened?]
(3) Reasoning: [Based on what you see in the generated simulation, explain how you arrive at the answer]
(4) Final Answer: [exact answer format requested in question]

**CRITICAL:** 
- FIRST generate the physics simulation image
- THEN carefully analyze what the generated image shows
- FINALLY provide your answer based on analyzing the generated image'''


LOGIC_SYSTEM_PROMPT = '''You are a helpful AI assistant. You need to think about the given prompt/question and any hints provided, then generate USEFUL VISUAL AIDS based on the hints during your thinking process, and finally answer the question based on your analysis and the generated images.

IMPORTANT REQUIREMENTS:
1. You MUST generate images that are USEFUL VISUAL AIDS for solving the problem (e.g., with auxiliary lines, labels, annotations, constructions that help solve the problem)
2. Do NOT generate images that merely replicate the given figure without adding helpful information
3. You MUST provide a final answer after your thinking process

Enclose your thinking process within <think> </think> tags, generate relevant images during thinking, then provide your final answer.

Format your output as: 
<think> 
Step 1: Analyze what auxiliary constructions would help solve this problem.
Step 2: Generate the visual aid with those constructions.
[generate USEFUL images with helpful additions like auxiliary lines, labels, or constructions]
Step 3: OBSERVE the generated image carefully and use the visual information to perform your reasoning.
Step 4: Based on what you see in the generated image, work through the solution.
</think>

Final Answer: [your answer based on the generated images and analysis]

REMEMBER: The generated images must be USEFUL VISUAL AIDS that add value beyond the original figure.'''


JIGSAW_SYSTEM_PROMPT = '''You are a helpful AI assistant solving visual jigsaw puzzles. You need to analyze the puzzle image with a gray box covering part of it, then generate a completed image by filling in the missing area, and finally select the correct option.

IMPORTANT REQUIREMENTS:
1. You MUST first generate a completed image by filling in the missing area covered by the gray box
2. Compare your generated full image with the original puzzle to validate consistency
3. Use the generated complete image to determine which option (A, B, C, or D) correctly fills the missing area
4. Do NOT answer before generating the completed image

Enclose your thinking process within <think> </think> tags, generate the completed image during thinking, then provide your final answer.

Format your output as: 
<think> 
Step 1: Analyze what is visible in the puzzle and what patterns/objects might be in the missing area.
Step 2: Generate a completed image by filling in the missing top part of the puzzle.
[Generate the full completed image]
Step 3: Carefully observe your generated complete image and compare it with the original puzzle to ensure consistency.
Step 4: Compare your generated complete image with each option (A, B, C, D) to find which one matches the missing area in your generated image.
</think>

Final Answer: [A/B/C/D]'''


MULTI_VIEW_SYSTEM_PROMPT = '''You are a helpful AI assistant analyzing multi-view images to determine camera movement direction. Given two images taken from different camera positions around the same scene, you need to reason about the spatial relationships and determine the camera rotation direction.

IMPORTANT REQUIREMENTS:
1. You MUST first generate a wider-angle image taken from farther away that includes ALL objects visible in both Image 1 and Image 2
2. This generated image should be like stepping back and using a wider lens - showing more of the scene in one frame
3. Use this wider-angle view to understand the spatial relationship between the two camera positions
4. Determine if the camera rotated counterclockwise (left) or clockwise (right) from Image 1 to Image 2
5. Do NOT answer before generating the wider-angle image

Think of it like this: If you step back from the scene and take a photo with a wider-angle lens, you can see all the objects from both viewpoints in one image.

Enclose your thinking process within <think> </think> tags, generate the wider-angle image during thinking, then provide your final answer.

Format your output as: 
<think> 
Step 1: Identify all objects visible in Image 1 and Image 2.
Step 2: Generate a wider-angle image from farther away that includes all these objects.
[Generate the wider-angle image showing the complete scene]
Step 3: Use this wider view to understand where the two cameras were positioned.
Step 4: Determine the rotation direction: counterclockwise (left) or clockwise (right)?
</think>

Final Answer: [left/right]'''



def get_system_prompt(problem_type):
    """Return corresponding system prompt based on problem type"""
    if problem_type == 'embodied':
        return EMBODIED_SYSTEM_PROMPT
    elif problem_type == 'physical':
        return PHYSICAL_SYSTEM_PROMPT
    elif problem_type == 'logical':
        return LOGIC_SYSTEM_PROMPT
    elif problem_type == 'jigsaw':
        return JIGSAW_SYSTEM_PROMPT
    elif problem_type == 'multi-view':
        return MULTI_VIEW_SYSTEM_PROMPT
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")


def file_exists_and_valid(png_path, txt_path):
    """Check if file exists and is valid"""
    return os.path.exists(png_path) and os.path.getsize(png_path) > 0

def process_benchmark(max_samples=None, output_dir="./gen_banana_tg"):
    """Process benchmark dataset
    
    Args:
        max_samples: Maximum number of samples to process, None means process all
        output_dir: Output directory path
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dataset
        print("Loading dataset cheryyunl/ROVER ROVER-TG...")
        dataset = load_dataset("cheryyunl/ROVER", "ROVER-TG")
        
        # Debug: View dataset structure
        print("Dataset keys:", list(dataset.keys()))
        
        # Assume data is in the default split
        data_split = dataset['train'] if 'train' in dataset else dataset[list(dataset.keys())[0]]
        
        print(f"Dataset loaded, total {len(data_split)} samples")
        
        # Show basic dataset information
        if len(data_split) > 0:
            first_item = data_split[0]
            print("Sample fields:", list(first_item.keys()))
            print("Problem type of first sample:", first_item.get('problem_type'))
        
        # Count different task types
        problem_types = {}
        for item in data_split:
            ptype = item.get('problem_type', 'unknown')
            problem_types[ptype] = problem_types.get(ptype, 0) + 1
        print("Task type distribution:", problem_types)
        
        # Limit processing quantity
        if max_samples is not None:
            total_to_process = min(max_samples, len(data_split))
            print(f"Limiting to first {max_samples} samples, actually processing {total_to_process}")
        else:
            total_to_process = len(data_split)
            print(f"Processing all {total_to_process} samples")
        
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        # HuggingFace Dataset needs select instead of slicing
        if max_samples is not None:
            items_to_process = data_split.select(range(total_to_process))
        else:
            items_to_process = data_split
        
        for idx, item in enumerate(items_to_process):
            try:
                # Get basic information
                task_id = item['id']
                problem_type = item.get('problem_type', 'unknown')
                image = item['image']  # PIL Image (main image)
                prompt = item['prompt']
                
                # Get second image based on task type (Image 2)
                # embodied: use example_image
                # physical: use image2 (if available)
                second_image = None
                image2_type = None
                if problem_type == 'embodied':
                    second_image = item.get('example_image')
                    image2_type = "example_image"
                elif problem_type == 'physical':
                    second_image = item.get('image2')
                    image2_type = "image2"
                elif problem_type == 'multi-view':
                    second_image = item.get('image2')
                    image2_type = "image2"
                else:
                    # Try both
                    second_image = item.get('example_image') or item.get('image2')
                    image2_type = "example_image or image2"
                
                # Generate output file paths
                png_path = os.path.join(output_dir, f"gen_{task_id}.png")
                txt_path = os.path.join(output_dir, f"gen_{task_id}.txt")
                
                # Check if valid output files already exist
                if file_exists_and_valid(png_path, txt_path):
                    print(f"[{idx+1}/{total_to_process}] Skipping {task_id} ({problem_type}): Output file already exists")
                    skipped_count += 1
                    continue
                
                print(f"[{idx+1}/{total_to_process}] Processing {task_id} ({problem_type})...")
                if second_image is not None:
                    print(f"  Includes second image ({image2_type})")
                
                # Get corresponding system prompt based on problem type
                system_prompt = get_system_prompt(problem_type)
                
                # Build complete prompt
                full_prompt = f"{system_prompt}\n\nTask: {prompt}"
                
                # Attempt to generate content (with retry mechanism)
                max_retries = 2
                generated_image = None
                all_text = ""
                
                for attempt in range(max_retries + 1):
                    try:
                        # Build multimodal input: main image + optional second image
                        contents = [full_prompt, image]
                        if second_image is not None:
                            contents.append(second_image)

                        response = client.models.generate_content(
                            model="gemini-2.5-flash-image-preview",
                            contents=contents,
                        )
                        
                        # Reset variables (for current attempt)
                        generated_image = None
                        all_text = ""
                        
                        # Process all parts in the response
                        for part in response.candidates[0].content.parts:
                            if part.text is not None:
                                all_text += part.text
                            elif part.inline_data is not None:
                                generated_image = Image.open(BytesIO(part.inline_data.data))
                        
                        # If image generated, break retry loop
                        if generated_image is not None:
                            break
                        elif attempt < max_retries:
                            print(f"  Attempt {attempt+1} did not generate an image, retrying...")
                            time.sleep(2)
                        
                    except Exception as e:
                        print(f"  Error generating content (attempt {attempt+1}): {e}")
                        if attempt < max_retries:
                            time.sleep(2)
                        else:
                            raise e
                
                # If still no image generated, add warning mark
                if generated_image is None:
                    print(f"  Warning: No image generated after {max_retries+1} attempts")
                    print(f"  Returned text: {all_text[:200]}...")
                    all_text = f"[WARNING: No image generated]\n\n{all_text}"
                
                # Save generated image and text
                if generated_image is not None:
                    generated_image.save(png_path)
                    print(f"  Image saved: {png_path}")
                    
                    # Save all text
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(all_text)
                    print(f"  Text saved: {txt_path}")
                    
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
    parser = argparse.ArgumentParser(description='ROVER-TG Benchmark Evaluation')
    parser.add_argument('--max-samples', type=int, default=10, 
                       help='Maximum number of samples to process (default: 10, set to 0 or negative for all)')
    parser.add_argument('--output-dir', type=str, default='./gen_banana_tg',
                       help='Output directory (default: ./gen_banana_tg)')
    
    args = parser.parse_args()
    
    # Set processing quantity
    max_samples = args.max_samples if args.max_samples > 0 else None
    
    print("ROVER-TG Benchmark Evaluation")
    print("=" * 50)
    print(f"Dataset: cheryyunl/ROVER ROVER-TG")
    print(f"Output directory: {args.output_dir}")
    
    if max_samples:
        print(f"Limiting processing count: {max_samples}")
    else:
        print("Processing all samples")
    
    process_benchmark(max_samples=max_samples, output_dir=args.output_dir)
