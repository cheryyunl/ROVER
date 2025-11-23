# Copyright (c) 2025 VortexBench Team
# SPDX-License-Identifier: Apache-2.0

# ROVER Evaluation Prompts
# 2 core metrics: interleaved_reasoning, reasoning_alignment
# 5 problem types: physical, embodied, logical, jigsaw, multi-view

# ============================================================================
# VISUAL REASONING PROMPTS (Compare reasoning_image vs generated image)
# ============================================================================

prompt_interleaved_reasoning_physical = """
You are evaluating the **visual reasoning quality** for a PHYSICAL problem (physics simulation).

## Task Understanding
Physical problems require the model to generate:
- A **simulation result image** showing what happens after physics plays out
- Starting from an initial state, predict the final state after physical interactions

## What You'll Receive
- **Problem Prompt**: {prompt}
- **Ground Truth Answer**: {answer}
- **Image 1**: Original image (initial state before simulation)
- **Image 2** (if available): Ground Truth result image (correct final state after physics)
- **Image 3**: Generated Image by the model (predicted final state to evaluate)

**Note**: Some problems may not have Image 2 (GT result). In that case, evaluate based on Image 1 and physical principles.

## Evaluation Criteria

**Your task**: 
- **If Image 2 (GT) is available**: Compare Image 3 (model's prediction) against Image 2 while referencing Image 1 (initial state)
- **If Image 2 (GT) is NOT available**: Evaluate whether Image 3 is a physically plausible result from Image 1 based on the problem prompt and physics principles

- **Image 1 shows**: Initial state before physics simulation
- **Image 2 shows** (if available): Ground truth final state (correct answer)
- **Image 3 shows**: Model's predicted final state (to evaluate)
- **Key question**: Does Image 3 show a physically plausible outcome?

You need to assess whether the generated image (Image 3) demonstrates **reasonable physical simulation**:

### For Physics Simulation Tasks:
High Quality (Score 4-5):
- Simulation result shows physically plausible outcome
- Key physical effects are captured (collision, gravity, flow, etc.)
- Result closely matches what Image 2 (GT) shows should happen
- Generated image clearly depicts the final state after physics plays out
- Objects' positions and states make physical sense

Medium Quality (Score 3):
- Simulation shows some physical understanding but with errors
- Some physical effects are captured, others missed
- Result partially aligns with Image 2 (GT) but has discrepancies
- Final state is somewhat plausible but not fully accurate

Poor Quality (Score 2):
- Simulation shows major errors but at least attempted
- Many physical effects missed or incorrect
- Result significantly differs from Image 2 (GT)
- Shows some understanding but with critical flaws

Failed (Score 1):
- Simulation result is physically implausible or completely wrong
- Violates basic physics laws entirely
- Result contradicts what should happen
- No clear demonstration of simulation or reasoning
- Image is not a valid simulation result
- **When in doubt between 1 and 2, prefer giving 1 if the image is clearly wrong**

## Scoring Guidelines with Examples

**5 - Excellent**: **ONLY for nearly perfect results** that match Image 2 (GT) almost identically
- Example: Ball drops problem → Image 3 shows ball in correct pit, correct orientation, all physics effects visible, **matches Image 2 perfectly** with at most tiny cosmetic differences
- **Reserve 5 points for exceptional quality only**
- **If you see any noticeable difference from GT, give 4 at most**

**4 - Good**: Most aspects correct with 1-2 minor issues
- Example: Ball drops problem → Image 3 shows ball in correct pit, slight position offset, most physics effects correct, closely matches Image 2

**3 - Adequate**: Majority correct but noticeable problems
- Example: Ball drops problem → Image 3 shows ball in approximately correct location, some physics effects missing, partially matches Image 2 but with visible differences

**2 - Poor**: Major errors but shows some attempt
- Example: Ball drops problem → Image 3 shows ball in wrong location but at least attempted simulation, has some physics understanding but critically flawed
- **Only give 2 if there was a genuine attempt with some correctness**

**1 - Failed**: Completely wrong, physically impossible, or not a valid result
- Example: Ball drops problem → Image 3 shows completely implausible outcome, severe physics violations, no resemblance to Image 2, or not a valid simulation image
- **IMPORTANT: When the image is clearly wrong or doesn't address the task, give 1 point**
- **When in doubt between 1 and 2, prefer 1 if the image is fundamentally incorrect**

## Evaluation Checklist (must check all items)

### For Physics Simulation Tasks:
Score each item as ✓ (correct), ✗ (incorrect), or ~ (partially correct):
- [ ] **Key Objects Present**: Are all important objects visible in final state? (✓ = all present, ~ = most present, ✗ = missing key objects)
- [ ] **Object Positions**: Are final positions physically plausible? (✓ = plausible, ~ = somewhat plausible, ✗ = implausible)
- [ ] **Physical Effects**: Are relevant physics captured (gravity/collision/flow)? (✓ = all captured, ~ = some captured, ✗ = ignored)
- [ ] **Final State Correctness**: Does it match what should happen? (✓ = matches GT, ~ = partially matches, ✗ = contradicts GT)
- [ ] **Physical Laws**: Are basic physics laws respected? (✓ = all respected, ~ = minor violations, ✗ = major violations)
- [ ] **Clarity**: Is the result state clear and unambiguous? (✓ = clear, ~ = somewhat clear, ✗ = unclear)

**Scoring Formula for Physics**: 
- 6 ✓ = 5 points
- 5 ✓ + 1 ~ = 4 points  
- 4 ✓ + 1-2 ~ = 3 points
- 2-3 ✓ + 2-3 ~ = 2 points
- <2 ✓ = 1 point

## Important Notes
- **If Image 2 (GT) is available**: Compare Image 3 against Image 2 as the primary reference
- **If Image 2 (GT) is NOT available**: Judge based on Image 1 (initial state) and physics principles
- **5 points ONLY for nearly perfect results** - reserve for exceptional quality
- **3 points requires majority correctness** - don't give 3 points for barely acceptable work
- **CRITICAL: Any violation of basic physics laws = maximum 2 points (or 1 if severe)**
- **When the image is clearly wrong, give 1 point**
- **When in doubt between 1 and 2, prefer 1 if the image is fundamentally incorrect**
- Use the checklist to ensure objective scoring
- Focus on whether the generated simulation is **physically plausible and matches the expected outcome**

Return JSON format:
{{
    "interleaved_reasoning_score": <1-5>,
    "reasoning": "<Use the checklist above. List each item with ✓/✗/~, then explain the overall quality and final score>"
}}
"""

prompt_interleaved_reasoning_embodied = """
You are evaluating the **visual reasoning quality** for an EMBODIED/ROBOTICS problem (trajectory planning).

## Task Understanding
Embodied problems require the model to generate:
- A **trajectory visualization** showing 10 waypoints overlaid on the scene
- Waypoints are typically marked with numbers or symbols
- Connected by lines showing the path
- Helps visualize the robot's planned motion from start to goal

## What You'll Receive
- **Problem Prompt**: {prompt}
- **Ground Truth Answer**: {answer}
- **Image 1**: Ground Truth Reasoning Image (shows what correct trajectory visualization should look like)
- **Image 2**: Generated Image by the model (the trajectory to evaluate)

## Evaluation Criteria

**Your task**: Evaluate if Image 2 (generated) demonstrates reasonable trajectory planning.

- **Primary reference**: Image 1 (ground truth) - shows one correct approach
- **Important**: Image 2 doesn't need to be identical to Image 1, but should be comparable in quality
- **If no Image 1 provided**: Judge based on the problem prompt and robotics principles
- **Key question**: Is Image 2 helpful for executing the robot task?

You need to assess whether the generated trajectory (Image 2) demonstrates **reasonable and helpful visual reasoning**:

### For Robotics/Embodied Tasks:
High Quality trajectory should have:
- Trajectory waypoints clearly visible and well-positioned
- Waypoints form a smooth, logical path
- Trajectory avoids obstacles and follows task requirements
- Visualization style is clear (markers, lines, labels)
- Generated trajectory is plausible and reasonable

Medium Quality trajectory may have:
- Waypoints present but placement could be better
- Path is somewhat logical but has issues (jerky motion, suboptimal routing)
- Some deviation from ideal but not completely wrong

Poor Quality trajectory shows:
- Waypoints missing or poorly positioned
- Path is illogical or violates task constraints
- Significant issues with obstacle avoidance or smoothness

## Scoring Guidelines with Examples

**5 - Excellent Visual Reasoning**
**ONLY for nearly perfect results** with all key elements correct and matching GT closely.

Robotics Example (Score 5):
- All 10 waypoints clearly visible with proper labels
- Smooth, continuous trajectory line connecting all points
- Path completely avoids all obstacles
- Reaches exact goal location
- Professional visualization quality
- **Matches ground truth approach almost perfectly**
- **Reserve 5 points for exceptional quality only**

**4 - Good Visual Reasoning**
Strong performance with only 1-2 minor issues.

Robotics Example (Score 4):
- 9 out of 10 waypoints visible (1 missing or unclear)
- Trajectory mostly smooth with one small gap
- Path avoids obstacles but cuts close to one
- Reaches goal location
- Good visualization quality

**3 - Adequate Visual Reasoning**
Majority correct but with several noticeable problems. Still demonstrates substantial understanding.

Robotics Example (Score 3):
- 6-8 waypoints visible, 2-4 missing or unclear
- Trajectory has multiple gaps but general path visible
- Path avoids major obstacles but takes suboptimal route
- Gets close to goal but not exact position
- Visualization acceptable but could be clearer

**2 - Poor Visual Reasoning**
Major issues but shows some attempt at trajectory visualization.

Robotics Example (Score 2):
- Only 3-5 waypoints visible but at least attempted
- Trajectory severely broken or erratic
- Shows some understanding but critically flawed
- **Only give 2 if there was a genuine attempt**

**1 - Failed Visual Reasoning**
Complete failure, nothing is correct or not a valid trajectory.

CRITICAL: Any violation of basic robotics rules (e.g., collision with obstacles, no trajectory) = 1 point.

Robotics Example (Score 1):
- No trajectory visualization visible
- Random or no waypoints
- No coherent path
- Completely wrong or missing
- Unusable for task
- **Path collides with obstacles (violates task requirement)**
- **Doesn't reach goal location (fails task)**
- **IMPORTANT: When the image is clearly wrong, give 1 point**
- **When in doubt between 1 and 2, prefer 1 if the trajectory is fundamentally incorrect**

## Evaluation Checklist (must check all items)

### For Robotics/Embodied Tasks:
Score each item as ✓ (correct), ✗ (incorrect), or ~ (partially correct):
- [ ] **Waypoint Count**: Are 9-10 waypoints clearly visible and identifiable? (✓ = 9-10, ~ = 6-8, ✗ = <6)
- [ ] **Trajectory Continuity**: Are waypoints connected with continuous lines? (✓ = continuous, ~ = small gaps, ✗ = broken)
- [ ] **Obstacle Avoidance**: Does trajectory avoid collisions? (✓ = clear avoidance, ~ = close to obstacles, ✗ = collisions)
- [ ] **Task Completion**: Does trajectory reach the goal location? (✓ = reaches goal, ~ = near goal, ✗ = doesn't reach)
- [ ] **Smoothness**: Is the path smooth and logical? (✓ = smooth, ~ = somewhat jerky, ✗ = erratic)
- [ ] **Visualization Quality**: Are markers/labels clear and readable? (✓ = clear, ~ = acceptable, ✗ = unclear)

**Scoring Formula for Robotics**: 
- 6 ✓ = 5 points
- 5 ✓ + 1 ~ = 4 points  
- 4 ✓ + 1-2 ~ = 3 points
- 2-3 ✓ + 2-3 ~ = 2 points
- <2 ✓ = 1 point

## Important Notes
- **Reference Image 1 (ground truth)** as a quality benchmark, but Image 2 doesn't need to be identical
- **If Image 1 is available**: Compare approach and quality; different trajectories can both be correct
- **If Image 1 is not available**: Judge based on problem prompt and robotics principles
- **5 points ONLY for nearly perfect results** - reserve for exceptional quality
- **3 points requires majority correctness** - don't give 3 points for barely acceptable work
- **CRITICAL: Any violation of basic robotics rules (collision, no trajectory) = 1 point**
- **When the trajectory is clearly wrong, give 1 point**
- **When in doubt between 1 and 2, prefer 1 if the trajectory is fundamentally incorrect**
- Use the checklist to ensure objective scoring
- Focus on whether the generated visualization is **helpful and reasonable for the robot task**

Return JSON format:
{{
    "interleaved_reasoning_score": <1-5>,
    "reasoning": "<Use the checklist above. List each item with ✓/✗/~, then explain the overall quality and final score>"
}}
"""

prompt_interleaved_reasoning_logical = """
You are evaluating the **visual reasoning quality** for a LOGICAL/MATHEMATICAL problem (typically geometry).

## Task Understanding
Logical problems require the model to generate a **useful visual aid** that includes:
- Auxiliary lines, constructions, or geometric relationships
- Angle marks, labels, or annotations
- Visual elements that help solve the problem
- NOT just a replication of the original figure, but meaningful additions

## What You'll Receive
- **Problem Prompt**: {prompt}
- **Ground Truth Answer**: {answer}
- **Image 1**: Original Problem Image (shows the initial geometry problem figure)
- **Image 2**: Ground Truth Reasoning Image (shows what correct auxiliary constructions look like)
- **Image 3**: Generated Image by the model (the visual aid to evaluate)

## Evaluation Criteria

**Your task**: **Strictly compare Image 3 (generated) against Image 2 (ground truth)** while referencing Image 1 (problem).

**CRITICAL REQUIREMENT**: If Image 3 uses a completely different approach from Image 2, give 1 point.
**CRITICAL**: If Image 3 is clearly wrong or doesn't match GT constructions, prefer giving 1 point.

You need to assess whether the generated image (Image 3) demonstrates similar auxiliary constructions to Image 2:

### High Quality Visual Aid (Score 4-5):
- Includes nearly all auxiliary constructions shown in Image 2 (GT)
- Construction approach closely matches Image 2
- Same key lines, circles, perpendiculars, or other geometric elements
- Clearly labeled or annotated where helpful
- May differ slightly in visual style but same mathematical strategy
- Demonstrates deep geometric understanding matching GT

### Medium Quality Visual Aid (Score 3):
- Includes majority of key constructions from Image 2 (GT)
- Construction approach generally matches Image 2
- Has most main auxiliary elements but missing 1-2 secondary ones
- Overall strategy aligns with GT
- Labels are adequate
- Shows solid geometric understanding similar to GT

### Poor Quality Visual Aid (Score 2):
- Only includes some constructions from Image 2, missing many key elements
- Approach partially aligns with Image 2 but significant gaps
- OR uses a completely different approach from Image 2
- Missing multiple key elements needed to solve
- Adds limited value

### Failed Visual Aid (Score 1):
- No meaningful auxiliary constructions
- Simply replicates Image 1 without adding value
- Constructions are irrelevant or mathematically incorrect
- Does not help solve the problem at all
- Visual aid is misleading or confusing

## Scoring Guidelines with Examples

**5 - Excellent**: **ONLY for nearly perfect results** that match GT approach almost identically
- Example: GT shows 3 auxiliary lines + angle marks → Generated has all 3 lines, all angle marks, clear labels, **almost identical to GT**
- **Reserve 5 points for exceptional quality that matches GT perfectly**
- **If approach differs noticeably from GT, give 4 at most**

**4 - Good**: Most key constructions, matches GT approach with minor gaps
- Example: GT shows 3 auxiliary lines + angle marks → Generated has all 3 lines, most angle marks, 1-2 labels missing
- **Clearly follows GT strategy with small differences**

**3 - Adequate**: Majority of key constructions, follows GT approach but incomplete
- Example: GT shows 3 auxiliary lines + angle marks → Generated has 2 out of 3 lines, some angle marks, partial labels
- Still follows GT strategy but missing some elements

**2 - Poor**: Only some elements present but shows some attempt
- Example: GT shows 3 auxiliary lines + angle marks → Generated has only 1 line, attempted but missing most key constructions
- **Only give 2 if there was a genuine attempt with some correctness**

**1 - Failed**: No useful constructions, completely wrong, or uses completely different approach
- Example: No auxiliary lines added, just copied Image 1, added irrelevant constructions, or uses totally different construction strategy from GT
- **CRITICAL: Completely different approach from GT = 1 point**
- **IMPORTANT: When the image clearly doesn't match GT approach or is wrong, give 1 point**
- **When in doubt between 1 and 2, prefer 1 if the constructions are fundamentally incorrect or missing**

## Evaluation Checklist (must check all items)

Score each item as ✓ (correct/present), ✗ (incorrect/missing), or ~ (partially correct):
- [ ] **Key Auxiliary Lines**: Are the main auxiliary lines/constructions present? (✓ = all key lines, ~ = most key lines, ✗ = missing major lines)
- [ ] **Geometric Correctness**: Are constructions geometrically sound and accurate? (✓ = all correct, ~ = minor errors, ✗ = major errors)
- [ ] **Angle/Length Marks**: Are important angles or lengths marked appropriately? (✓ = well marked, ~ = some marks, ✗ = no marks)
- [ ] **Labels**: Are key points, lines, or angles labeled clearly? (✓ = clear labels, ~ = some labels, ✗ = no/poor labels)
- [ ] **Problem Relevance**: Do constructions directly help solve the problem? (✓ = highly relevant, ~ = somewhat relevant, ✗ = irrelevant)
- [ ] **Adds Value**: Does it go significantly beyond Image 1 (problem)? (✓ = substantial additions, ~ = some additions, ✗ = minimal additions)

**Scoring Formula**: 
- 6 ✓ = 5 points
- 5 ✓ + 1 ~ = 4 points  
- 3-4 ✓ + 1-2 ~ = 3 points
- 1-2 ✓ + 2-3 ~ = 2 points
- <1 ✓ = 1 point

## Important Notes
- **MUST strictly compare Image 3 (generated) with Image 2 (ground truth)** while referencing Image 1 (problem)
- **If completely different approach from GT = 1 point**
- **If Image 3 is clearly wrong or missing key constructions = 1 point**
- **3 points requires majority of key constructions matching GT** - not just "some" constructions
- **5 points ONLY for nearly perfect match with GT** - reserve for exceptional quality
- Check: Does Image 3 follow similar construction strategy as Image 2?
- Auxiliary constructions should be mathematically sound and match GT approach
- **When in doubt between 1 and 2, prefer giving 1 point**
- Use the checklist above for objective scoring

Return JSON format:
{{
    "interleaved_reasoning_score": <1-5>,
    "reasoning": "<Use the checklist above. List each item with ✓/✗/~, compare with Image 2 (GT), and explain the final score>"
}}
"""

prompt_interleaved_reasoning_jigsaw = """
You are evaluating the **visual reasoning quality** for a JIGSAW problem (Perception).

## Task Understanding
Jigsaw puzzle tasks require the model to:
- A gray box covers part of the original image
- Generate a **completed image** by filling in the missing area
- The completion should be visually coherent with the visible parts

## What You'll Receive
- **Problem Prompt**: {prompt}
- **Ground Truth Answer**: {answer}
- **Image 1**: Ground Truth complete image (correct completion)
- **Image 2**: Model's generated complete image (to evaluate)

## Evaluation Criteria

**Your task**: Compare Image 2 (model's completion) with Image 1 (GT completion).

You need to assess whether the generated completion (Image 2) demonstrates **reasonable perceptual reasoning**:

### For Jigsaw Puzzle Tasks:
High Quality (Score 4-5):
- Generated complete image is visually coherent and natural
- Missing area is filled consistently with visible parts
- Patterns, colors, textures flow smoothly from visible to generated areas
- Objects and structures in generated area are plausible
- Generated completion closely matches Image 1 (GT)
- Image looks like a realistic, complete picture

Medium Quality (Score 3):
- Generated complete image has some coherence but with noticeable issues
- Completion somewhat matches visible parts but has inconsistencies
- Some patterns or objects don't align perfectly
- Partially plausible but has visible artifacts or errors
- Moderately aligns with Image 1 (GT)

Poor Quality (Score 2):
- Generated image has major coherence issues
- Completion poorly matches visible parts
- Many patterns or objects are inconsistent
- Contains significant artifacts or errors
- Significantly differs from Image 1 (GT)

Failed (Score 1):
- Generated image is incoherent or doesn't complete the puzzle
- Missing area is filled with random or irrelevant content
- No continuity with visible parts of the puzzle
- Completely implausible or nonsensical completion
- Does not demonstrate understanding of the visual pattern

## Scoring Guidelines with Examples

**5 - Excellent**: **ONLY for nearly perfect completions** that match GT almost identically
- Example: Scene with gray box on top → Model generates perfectly matching sky/building top that blends seamlessly with visible bottom part, **matches GT completion almost perfectly**
- **Reserve 5 points for exceptional quality only**
- **If you see noticeable differences from GT, give 4 at most**

**4 - Good**: Completion is coherent with 1-2 minor issues
- Example: Completion mostly matches but one small area has slight color mismatch or minor texture discontinuity

**3 - Adequate**: Completion is acceptable but with noticeable problems
- Example: Completion has correct general structure but several areas show visible seams, color differences, or pattern breaks

**2 - Poor**: Completion has major issues but shows some attempt
- Example: Attempted completion but has wrong structures, significant issues
- **Only give 2 if there was a genuine attempt with some correctness**

**1 - Failed**: Completion is completely wrong, incoherent, or not a valid completion
- Example: Generates random patterns, completely different scene, nonsensical content, or no meaningful completion
- **CRITICAL: If model just copied the original image without actually filling the gray box area, give 1 point**
- **IMPORTANT: When the completion is clearly wrong or doesn't make sense, give 1 point**
- **When in doubt between 1 and 2, prefer 1 if the completion is fundamentally incorrect**

## Evaluation Checklist (must check all items)

Score each item as ✓ (correct), ✗ (incorrect), or ~ (partially correct):
- [ ] **Actually Filled Missing Area (CHECK FIRST)**: Did the model genuinely fill the gray box area with new content, or just copy the original? (✓ = filled with appropriate new content, ~ = attempted but poor quality, ✗ = just copied original/no real filling → give 1 point)
- [ ] **Visual Coherence**: Does the completion blend naturally with visible parts? (✓ = seamless, ~ = minor seams, ✗ = obvious breaks)
- [ ] **Pattern Continuity**: Do patterns/textures continue smoothly? (✓ = smooth, ~ = some breaks, ✗ = discontinuous)
- [ ] **Color Consistency**: Are colors consistent across the boundary? (✓ = consistent, ~ = slight mismatch, ✗ = wrong colors)
- [ ] **Object Plausibility**: Are generated objects/structures plausible? (✓ = plausible, ~ = somewhat plausible, ✗ = implausible)
- [ ] **Matches GT**: Does it match Image 1 (GT) completion approach? (✓ = close match, ~ = partial match, ✗ = very different)

**Scoring Formula**: 
- Didn't fill missing area (just copied original) → 1 point
- 5 ✓ = 5 points
- 4 ✓ + 1 ~ = 4 points  
- 3 ✓ + 1-2 ~ = 3 points
- 1-2 ✓ + 2-3 ~ = 2 points
- <1 ✓ = 1 point

## Important Notes
- **CRITICAL FIRST CHECK**: Did the model genuinely fill the missing gray box area? If just copied the original image without filling, give 1 point
- **Compare Image 2 (model completion) with Image 1 (GT completion)** as the primary reference
- **5 points ONLY for nearly perfect completions** - extremely rare, reserve for exceptional quality that matches GT almost identically
- **"Matches GT" item requires VERY close match** - if you see noticeable structural/content differences from GT, mark as ~ or ✗, not ✓
- **Be strict with checklist** - don't give ✓ unless genuinely correct/seamless, use ~ for "good enough"
- **3 points requires majority correctness** - most aspects should be coherent
- **When the completion is clearly wrong or incoherent, give 1 point**
- **When in doubt between scores, prefer lower score**
- Focus on visual coherence, pattern continuity, and naturalness
- Use the checklist to ensure objective scoring

Return JSON format:
{{
    "interleaved_reasoning_score": <1-5>,
    "reasoning": "<Use the checklist above. List each item with ✓/✗/~, compare with Image 1 (GT), and explain the final score>"
}}
"""

prompt_interleaved_reasoning_multi_view = """
You are evaluating the **visual reasoning quality** for a MULTI-VIEW problem (Perception).

## Task Understanding
Multi-view perception problems require the model to:
- Receive **two images from different camera positions** around the same scene
- Generate a **wider-angle image** from farther away showing all objects from both views
- This wider view helps understand the spatial relationship and camera rotation
- Then determine if camera rotated clockwise or counter-clockwise

## What You'll Receive
- **Problem Prompt**: {prompt}
- **Ground Truth Answer**: {answer}
- **Image 1**: First camera view
- **Image 2**: Second camera view
- **Image 3**: Model's generated wider-angle view (to evaluate)

## Evaluation Criteria

**Your task**: Evaluate whether Image 3 (generated wider view) reasonably combines information from Images 1 and 2.

You need to assess whether the generated wider-angle view demonstrates **reasonable spatial reasoning**:

### High Quality (Score 4-5):
- All key objects from both views are visible in the wider view
- Spatial relationships between objects are consistent with both input views
- Objects are positioned correctly relative to each other
- View angle and distance are appropriate (wider and farther)
- Scene composition is coherent and natural
- Helps understand the camera rotation between the two views

### Medium Quality (Score 3):
- Most key objects from both views are visible
- Spatial relationships are mostly consistent
- Minor issues with positioning or view angle
- Generally helps understand the scene layout
- Aids in determining camera rotation

### Poor Quality (Score 2):
- Missing several key objects from the input views
- Spatial relationships are inconsistent or incorrect
- Significant positioning errors
- View angle doesn't help clarify the scene
- Difficult to determine camera rotation from this view

### Failed (Score 1):
- Missing most objects from the input views
- Completely incorrect spatial layout
- Contradicts information from input views
- Doesn't provide a coherent wider view
- Unusable for understanding camera rotation

## Scoring Guidelines with Examples

**5 - Excellent**: **ONLY for nearly perfect results** with all elements correct
- Example: Both input views show partial room scenes → Generated view shows complete room with all furniture in correct positions matching both partial views **perfectly**
- **Reserve 5 points for exceptional quality only**
- **If you see noticeable issues, give 4 at most**

**4 - Good**: Most elements correct with 1-2 minor issues
- Example: All major objects present and positioned correctly, one small object slightly off or one minor spatial inconsistency

**3 - Adequate**: Majority correct but noticeable gaps
- Example: Most objects present but 1-2 missing, spatial layout generally matches but some relationships unclear

**2 - Poor**: Major issues but shows some attempt
- Example: Attempted to create wider view but several objects missing or positioning significantly wrong
- **Only give 2 if there was a genuine attempt with some correctness**

**1 - Failed**: Unusable, completely wrong, or not a valid wider view
- Example: Most objects missing, spatial layout completely different from input views, incomprehensible wider view, or doesn't combine the two views meaningfully
- **CRITICAL: If model just copied Image 1 or Image 2 without actually generating a wider view combining both, give 1 point**
- **IMPORTANT: When the wider view is clearly wrong or doesn't make sense, give 1 point**
- **When in doubt between 1 and 2, prefer 1 if the view is fundamentally incorrect**

## Evaluation Checklist (must check all items)

Score each item as ✓ (correct), ✗ (incorrect), or ~ (partially correct):
- [ ] **Actually Generated Wider View (CHECK FIRST)**: Did the model genuinely create a wider view combining both views, or just copied Image 1 or Image 2? (✓ = wider view combining both, ~ = attempted but poor, ✗ = just copied one view/no real combination → give 1 point)
- [ ] **Object Completeness**: Are all key objects from both input views present? (✓ = all present, ~ = most present, ✗ = many missing)
- [ ] **Spatial Consistency**: Do object positions match both input views? (✓ = fully consistent, ~ = mostly consistent, ✗ = contradictions)
- [ ] **Relative Positioning**: Are objects positioned correctly relative to each other? (✓ = correct, ~ = minor issues, ✗ = major errors)
- [ ] **View Angle**: Is the wider angle appropriate for showing both views? (✓ = appropriate, ~ = acceptable, ✗ = inappropriate)
- [ ] **Scene Coherence**: Does the wider view form a coherent scene? (✓ = coherent, ~ = somewhat coherent, ✗ = incoherent)

**Scoring Formula**: 
- Just copied one view (didn't generate wider view) → 1 point
- 5 ✓ = 5 points
- 4 ✓ + 1 ~ = 4 points  
- 3 ✓ + 1-2 ~ = 3 points
- 1-2 ✓ + 2-3 ~ = 2 points
- <1 ✓ = 1 point

## Important Notes
- **CRITICAL FIRST CHECK**: Did the model genuinely create a wider view combining both views? If just copied Image 1 or Image 2, give 1 point
- Focus on whether Image 3 successfully combines information from Images 1 and 2
- **5 points ONLY for nearly perfect results** - extremely rare, reserve for exceptional quality
- **Be strict with checklist** - don't give ✓ unless genuinely correct, use ~ for "good enough"
- **3 points requires majority correctness** - most objects and relationships should be correct
- **When the wider view is clearly wrong or doesn't make sense, give 1 point**
- **When in doubt between scores, prefer lower score**
- The generated view should be wider and show more context than either input view alone
- Spatial relationships must be consistent with BOTH input views
- Use the checklist to ensure objective scoring

Return JSON format:
{{
    "interleaved_reasoning_score": <1-5>,
    "reasoning": "<Use the checklist above. List each item with ✓/✗/~, describe consistency with input views, and explain the final score>"
}}
"""

# ============================================================================
# REASONING ALIGNMENT PROMPTS (Compare prompt + generated image + model answer)
# ============================================================================

prompt_reasoning_alignment_physical = """
You are evaluating the **reasoning alignment** for a PHYSICAL problem (robotics trajectory or physics simulation).

## Task Understanding
Physical problems require the model to:
1. For **Robotics/Embodied**: Generate a trajectory visualization with 10 waypoints overlaid on the scene
2. For **Physics Simulation**: Generate a simulation result image showing what happens after physics plays out

## What You'll Receive
- **Problem Prompt**: {prompt}
- **Image 1**: Original image (initial state or scene - shows the problem)
- **Image 2**: Generated image (trajectory or simulation result)
- **Model's Answer Text**: {model_answer} (from the gen_xxx.txt file)

## Evaluation Criteria

You need to assess whether the model's answer is **grounded in and aligned with** the generated image:

### Strong Alignment (Score 4-5):
- The model explicitly references the generated image in its reasoning
- The answer directly corresponds to what's shown in the generated image
- For robotics: Answer coordinates match the visualized trajectory waypoints
- For physics: Answer reflects the simulated outcome shown in the generated image
- The model demonstrates it used the generated image to derive the answer

### Weak Alignment (Score 2-3):
- The model provides an answer but doesn't clearly reference the generated image
- Some connection exists but it's vague or inconsistent
- Answer partially matches the image but has discrepancies
- The model might have generated the image but didn't actively use it for reasoning

### No Alignment (Score 1):
- Answer contradicts what's shown in the generated image
- No evidence the model used the generated image to answer
- Answer appears to be derived independently without visual grounding
- For robotics: Coordinates don't match the visualized waypoints at all
- For physics: Answer doesn't reflect what the simulated image shows

## Scoring Guidelines

### **5 - Perfect Alignment**
**ONLY for exceptional cases** where the generated image is ESSENTIAL and makes a DECISIVE CONTRIBUTION to deriving the answer. The answer must be **impossible or much harder to obtain without the generated image**.

**Requirements for Score 5 (ALL must be met):**
1. Model explicitly references the generated image throughout reasoning (not just once)
2. Answer is directly extracted from analyzing specific visual details in the image
3. The generated image provides CRITICAL information that determines the answer
4. Reasoning demonstrates: generate → observe specific details → extract information → derive answer
5. **Without the generated image, the answer would be very difficult or impossible to obtain**

**Robotics Example (Score 5):**
- Reasoning: "Based on the trajectory I generated, I identified 10 waypoints. Waypoint 1 at (1541, 678) starts from the current gripper position, waypoint 2 at (1598, 633) moves toward the target..."
- All coordinates explicitly match the visualized waypoints
- **The trajectory visualization is ESSENTIAL - coordinates are read directly from the visual markers**
- Answer directly derived from analyzing the generated image

**Physics Example (Score 5):**
- Reasoning: "In the simulation result, I observe the ball has fallen into the left pit. The final position shows the ball at rest at the bottom of the left container..."
- Answer explicitly states what the generated image shows
- **The simulation image is DECISIVE - the answer comes from observing the final state**
- Describes specific object positions visible in the image
- Clear phrases like "the simulation shows," "from the generated result"

**CRITICAL: Don't give 5 just because the model mentions the image. Give 5 ONLY when the image makes a decisive contribution to the answer.**

---

### **4 - Good Alignment**
Answer shows **excellent grounding** in the image with detailed references. **Significantly better than just mentioning the image.**

**Requirements for Score 4:**
- Model references the generated image multiple times with specific details
- Answer is clearly derived from analyzing the image, not just mentioned alongside it
- Reasoning describes specific visual elements and how they lead to the answer
- Shows deep engagement with the generated image
- **Much more than just "I generated X and the answer is Y"**

**Robotics Example (Score 4):**
- Reasoning: "Looking at the trajectory, waypoint 1 at (1541, 678) starts from gripper, waypoint 2 moves toward the target avoiding the obstacle on the left, waypoint 3..."
- Describes the path in detail with multiple waypoint references
- Explains how the trajectory solves the task
- **Goes beyond basic mention - shows active analysis**

**Physics Example (Score 4):**
- Reasoning: "In my simulation, I observe the ball rolling down the slope, bouncing off the left wall, and settling in the left pit. The final position shows..."
- Describes multiple stages or details from the simulation
- Explains what specific aspects of the image show
- **Demonstrates thorough analysis, not just a brief mention**

**Note: If you're unsure between 4 and 5, ask: "Is the image ESSENTIAL and DECISIVE?" If not completely certain, give 4.**
**Note: If the model only briefly mentions the image or doesn't provide detailed analysis, give 3 at most.**

---

### **3 - Adequate Alignment**
**Basic/ordinary match** - model mentions the image and answer corresponds to it, but without detailed analysis. **This is the baseline for "mentions + matches".**

**What Score 3 looks like:**
- Model mentions generating/using the image
- Answer generally corresponds to what the image shows
- But lacks detailed description or thorough analysis
- Brief reference rather than deep engagement
- **"I generated X and got Y" style - mentions but doesn't deeply analyze**

**Robotics Example (Score 3):**
- Reasoning: "I generated a trajectory visualization. Based on this, the waypoints are: [lists coordinates]"
- Mentions trajectory but doesn't describe path details
- Coordinates somewhat match but not thoroughly explained
- **Basic mention + answer provided, but not detailed analysis**

**Physics Example (Score 3):**
- Reasoning: "I simulated the physics. The ball ends up in the left pit."
- Mentions simulation but doesn't describe what the image shows in detail
- Answer matches image but lacks specific visual details
- **Ordinary reference without thorough description**

**This is the score for ordinary/baseline alignment - just mentioning the image and having the answer match.**

---

### **2 - Poor Alignment**
Answer **barely relates** to image but shows some attempt at connection. **Also give 2 if the generated image itself has obvious errors.**

**Reasons for Score 2:**
1. Model vaguely mentions image but doesn't really use it for reasoning
2. Answer has weak connection to what the image shows
3. **IMPORTANT: If the generated image has obvious errors/quality issues, maximum score is 2** (even if the model references it)

**Robotics Example (Score 2):**
- Reasoning vaguely mentions "trajectory" but doesn't describe details
- Most coordinates don't match the visualized waypoints (>100px off)
- Answer seems mostly calculated independently but some reference to image
- **Or: The trajectory visualization has obvious errors (missing waypoints, collisions) but model still references it**

**Physics Example (Score 2):**
- Reasoning briefly mentions "simulation" but doesn't describe what it shows
- Answer has weak connection to the generated image
- **Or: The simulation image is clearly wrong (violates physics) but model still mentions it**
- Shows some attempt at alignment but mostly disconnected

**CRITICAL: If the generated image has obvious quality issues or errors, alignment is capped at 2 points, regardless of how much the model references it.**

---

### **1 - No Alignment**
Answer **completely ignores or contradicts** the generated image.

**CRITICAL: If answer contradicts the generated image = 1 point.**

**Robotics Example (Score 1):**
- Reasoning: "I calculated the optimal path using inverse kinematics..." (no mention of generated visualization)
- Coordinates completely different from visualized waypoints
- No reference to the generated image at all
- Answer derived purely from text/calculation

**Physics Example (Score 1):**
- Reasoning: "Based on the initial setup, the ball should..." (no mention of simulation)
- **Answer contradicts the generated simulation result** (e.g., says "right pit" when image shows "left pit")
- No mention of what the generated image actually shows
- Answer seems to be from pure reasoning without visual grounding
- **IMPORTANT: When answer contradicts the image, give 1 point**
- **When in doubt between 1 and 2, prefer 1 if there's clear contradiction or no reference**

## Evaluation Checklist (must check all items)

### For Robotics/Embodied Tasks:
- [ ] **Coordinate Matching (CHECK FIRST)**: Do answer coordinates ACTUALLY match visualized waypoint positions in the image? (✓ = close match <50px avg, ~ = rough match 50-100px, ✗ = no match >100px → If ✗, max 2 points)
- [ ] **Mentions Generated Image**: Does reasoning text explicitly mention "trajectory"/"visualization"/"generated image"? (✓ = explicit mention, ~ = implicit reference, ✗ = no mention)
- [ ] **Describes Visual Elements**: Does reasoning describe specific waypoints/markers visible in the image? (✓ = describes details, ~ = vague description, ✗ = no description)
- [ ] **Sequential Reference**: Does reasoning follow the trajectory sequence shown in the image? (✓ = follows sequence, ~ = partially follows, ✗ = ignores sequence)
- [ ] **Visual Analysis**: Does reasoning analyze the image to extract coordinates, not just mention it? (✓ = clear analysis, ~ = weak analysis, ✗ = no analysis)

**Scoring Formula for Robotics**: 
- Coordinate Matching ✗ → max 2 points (answer doesn't match image, no real alignment)
- Coordinate Matching ✓ + 4✓ alignment → 5 points
- Coordinate Matching ✓ + 3✓+1~ → 4 points
- Coordinate Matching ✓ + 2-3✓ → 3 points
- Coordinate Matching ~ + weak alignment → 2 points
- Contradicts image or no coordinates → 1 point

### For Physics Simulation Tasks:
- [ ] **Outcome Matching (CHECK FIRST)**: Does answer ACTUALLY match what the simulation image shows? (✓ = matches, ~ = partially matches, ✗ = contradicts → If ✗, max 2 points)
- [ ] **Mentions Generated Image**: Does reasoning text explicitly mention "simulation"/"generated result"? (✓ = explicit mention, ~ = implicit reference, ✗ = no mention)
- [ ] **Describes Final State**: Does reasoning describe the final state shown in the simulation image? (✓ = describes details, ~ = vague description, ✗ = no description)
- [ ] **Object Position Reference**: Does answer mention where key objects ended up (as shown in image)? (✓ = specific positions, ~ = general positions, ✗ = no positions)
- [ ] **Visual Analysis**: Does reasoning analyze the simulated image to derive the answer? (✓ = clear analysis, ~ = weak analysis, ✗ = no analysis)

**Scoring Formula for Physics**: 
- Outcome Matching ✗ → max 2 points (answer doesn't match image, no real alignment)
- Outcome Matching ✓ + 4✓ alignment → 5 points
- Outcome Matching ✓ + 3✓+1~ → 4 points
- Outcome Matching ✓ + 2-3✓ → 3 points
- Outcome Matching ~ + weak alignment → 2 points
- Contradicts image or no reference → 1 point

## Important Notes
- Check Coordinate/Outcome Matching FIRST - answer must match image content, not just mentioned
- Focus on whether the model used the generated image to derive the answer
- If unsure between scores, prefer lower score

**5 Points (Exceptional):**
- Answer matches image content AND image makes DECISIVE contribution
- Ask: "Could the model have given this answer without the generated image?" If yes, don't give 5

**4 Points (Excellent):**
- Answer matches image content
- Detailed analysis with multiple specific references to the image
- If only brief mention, give 3 at most

**3 Points (Adequate):**
- Answer matches image content
- Model mentions the image and answer corresponds to it
- Lacks detailed description or thorough analysis

**2 Points (Poor):**
- Answer doesn't match image content → Max 2 points
- OR: Weak connection to image
- OR: Image has obvious errors

**1 Point (None):**
- Answer contradicts the generated image
- When in doubt between 1 and 2, prefer 1

Return JSON format:
{{
    "reasoning_alignment_score": <1-5>,
    "reasoning": "<Use the checklist above. List each item with ✓/✗/~, then explain the alignment quality and final score>"
}}
"""

prompt_reasoning_alignment_logical = """
You are evaluating the **reasoning alignment** for a LOGICAL/MATHEMATICAL problem (typically geometry).

## Task Understanding
Logical problems require the model to:
1. Generate a **visual aid** with auxiliary constructions (auxiliary lines, angle marks, labels, annotations)
2. Observe and analyze the generated visual aid
3. Use the visual information to solve the problem and provide an answer

**CRITICAL for Geometry Problems:**
- Most geometry problems CAN be solved through pure algebraic/symbolic reasoning without visual aids
- The purpose of the visual aid is to provide INSIGHTS that make the solution easier or more intuitive
- **Just mentioning "I drew auxiliary lines" does NOT prove the model actually used them**
- **Be skeptical: Did the model truly gain insights from observing the visual aid, or just use algebra?**

## What You'll Receive
- **Problem Prompt**: {prompt}
- **Ground Truth Answer**: {answer}
- **Image 1**: Original problem image (shows the initial geometry problem)
- **Image 2**: Ground Truth Reasoning Image (shows what correct auxiliary constructions look like)
- **Image 3**: Generated image by the model (model's visual aid with auxiliary constructions - to evaluate)
- **Model's Answer Text**: {model_answer}

**IMPORTANT:** You now have access to Image 2 (GT reasoning image) to help judge if the model's generated image (Image 3) is reasonable or completely wrong.

## Evaluation Criteria

**STEP 1 - Check Generated Image Quality (Compare Image 3 with GT Image 2):**
- WRONG: Completely incorrect constructions, missing ALL key elements from GT, or just copies Image 1 → Give 1 point
- POOR: Different approach from GT, major errors, or missing most key constructions → Give 1 point (low quality image means model didn't truly use it, even if answer is correct)
- ADEQUATE: Has majority of key constructions from GT, approach matches → Can consider 3-5 points

**CRITICAL: Image quality determines maximum score. Wrong/Poor image = 1 point, regardless of text or answer correctness.**

**STEP 2 - Check Text Alignment (only if Image 3 is at least ADEQUATE):**

### Strong Alignment (Score 4-5):
- Model explicitly references the generated constructions in its reasoning
- Answer derived by observing and analyzing the visual aid
- Describes specific constructions, angles, or relationships from the image
- Clear evidence the model used the visual aid to solve the problem

### Weak Alignment (Score 2-3):
- Model mentions generating constructions but doesn't clearly analyze them
- Some connection but vague about using visual information
- Answer correct but unclear if from visual aid or pure calculation

### No Alignment (Score 1):
- No evidence model used the generated image
- Pure symbolic reasoning without visual grounding
- Answer contradicts the image or they seem disconnected

## Scoring Guidelines

**5 - Perfect Alignment**
- Image 3 is ADEQUATE + Answer is CORRECT
- Model explicitly references constructions throughout
- Visual aid is ESSENTIAL - answer derived FROM observing constructions

**4 - Good Alignment**
- Image 3 is ADEQUATE + Answer is CORRECT
- Model references specific constructions multiple times with details
- Goes beyond just "I drew X and the answer is Y"

**3 - Adequate Alignment**
- Image 3 is ADEQUATE + Answer is CORRECT
- Model mentions constructions and answer is correct
- Lacks detailed visual analysis

**2 - Poor Alignment**
- Image ADEQUATE + Answer CORRECT but model uses pure algebra (mentions constructions but doesn't analyze them)
- Weak alignment even with correct answer
- Note: Image POOR or Answer WRONG → 1 point (see scoring formula)

**1 - No Alignment**
- Image 3 is WRONG → Give 1 point
- OR: Image 3 is POOR (low quality) → Give 1 point (even if answer is correct, didn't truly use image)
- OR: No visual reference
- OR: Answer contradicts the image

## Evaluation Checklist

**STEP 1 - Image Quality (Compare Image 3 with GT Image 2):**
- [ ] **Image Quality**: ✓ = ADEQUATE (has majority key constructions, matches GT approach) / ~ = POOR (different approach, major errors) / ✗ = WRONG (completely incorrect, missing ALL key elements)

**STEP 2 - Answer Correctness:**
- [ ] **Answer Correctness**: ✓ = correct / ✗ = wrong

**STEP 3 - Text Alignment:**
- [ ] **Mentions Image**: Explicitly references constructions? (✓ = multiple times / ~ = once / ✗ = no mention)
- [ ] **Describes Elements**: Describes specific lines/angles from Image 3? (✓ = multiple elements / ~ = vague / ✗ = none)
- [ ] **Observational Language**: Uses "I observe," "from diagram"? (✓ = multiple / ~ = some / ✗ = none)
- [ ] **Visual Grounding**: Answer from visual insights or pure algebra? (✓ = visual / ~ = mixed / ✗ = pure algebra)

**Scoring Formula:**
- Image ✗ (WRONG) → 1 point (stop)
- Image ~ (POOR) → 1 point (stop, regardless of answer correctness)
- Image ✓ + Answer ✗ → 1 point (failed reasoning)
- Image ✓ + Answer ✓ + 4✓ alignment → 5 points
- Image ✓ + Answer ✓ + 3✓+1~ → 4 points
- Image ✓ + Answer ✓ + 2-3✓ → 3 points

## Important Notes
- Check Image 3 quality FIRST (compare with GT Image 2)
- **CRITICAL**: Image WRONG or POOR → 1 point (stop, regardless of answer)
- **CRITICAL**: Image ADEQUATE + Answer WRONG → 1 point (failed reasoning)
- Only Image ADEQUATE + Answer CORRECT can get 3-5 points
- Be skeptical for geometry - many solvable purely algebraically
- If unsure between scores, prefer lower score

Return JSON format:
{{
    "reasoning_alignment_score": <1-5>,
    "reasoning": "<Part 1: Image Quality [WRONG/POOR/ADEQUATE]. Part 2: Answer Correctness. Part 3: Alignment checklist with ✓/✗/~. Final score with justification.>"
}}
"""

prompt_reasoning_alignment_jigsaw = """
You are evaluating the **reasoning alignment** for a JIGSAW problem (Perception).

## Task Understanding
Jigsaw puzzle tasks require the model to:
- Original image has a gray box covering part of it
- Model generates a completed image by filling in the missing area
- Then selects which option (A, B, C, or D) correctly matches the completion
- The answer should be based on comparing the generated complete image with the options

## What You'll Receive
- **Problem Prompt**: {prompt}
- **Ground Truth Answer**: {answer}
- **Image 1**: Original image with gray box (shows the problem)
- **Image 2**: Model's generated completion
- **Model's Answer Text**: {model_answer}

## Evaluation Criteria

**CRITICAL: Check answer correctness AND consistency with generated image**

Common issues:
1. Answer is WRONG (doesn't match ground truth) → reasoning failed, regardless of image quality
2. Answer contradicts what the generated image shows → fundamental alignment failure

**STEP 1 - Answer Correctness (Check FIRST):**
- Compare model's answer with Ground Truth Answer
- **If answer is WRONG → Maximum 2 points** (reasoning failed to produce correct answer)
- Even if reasoning looks good, wrong answer = failed reasoning

**STEP 2 - Image-Answer Consistency:**
- **For puzzles**: Does the chosen option match what the completed image shows?
- **For multi-view**: Does the rotation direction match what the wider view indicates?
- **If answer contradicts the generated image → Give 1 point**

**STEP 3 - Alignment Depth (only if answer is correct AND consistent):**

### Strong Alignment (Score 4-5):
- Answer consistent with generated image
- Model explicitly describes visual details from the image
- For puzzles: Compares completion with each option based on visual features
- For multi-view: Analyzes spatial relationships to determine rotation
- Clear evidence answer derived by examining the image

### Weak Alignment (Score 2-3):
- Answer consistent with generated image
- Model mentions image but doesn't clearly analyze it
- Vague or unclear connection

### No Alignment (Score 1):
- **Answer contradicts generated image** (wrong option for puzzle, wrong rotation for multi-view)
- No evidence model used the generated image
- Answer appears guessed without visual grounding

## Scoring Guidelines

**5 - Perfect Alignment** (ONLY for exceptional cases)
- **Answer is CORRECT** (matches ground truth)
- Answer matches generated image
- Model explicitly references image with specific details
- Image is ESSENTIAL for determining answer

**4 - Good Alignment** (Detailed analysis)
- **Answer is CORRECT**
- Answer matches generated image
- Model references image multiple times with details

**3 - Adequate Alignment** (Basic mention + match)
- **Answer is CORRECT**
- Answer matches generated image
- Model mentions image and provides answer
- Example: "I completed the image. The answer is B."

**2 - Poor Alignment**
- **Answer is WRONG** (doesn't match ground truth) → Max 2 points
- OR: Answer correct but weak connection to image
- OR: Generated image has obvious errors

**1 - No Alignment**
- Answer contradicts generated image
- No visual reference at all
- Answer completely wrong with no grounding

## Evaluation Checklist (must check all items)

### For Jigsaw Puzzle Tasks:
- [ ] **Answer Correctness (CHECK FIRST)**: Does model's answer match Ground Truth? (✓ = correct / ✗ = wrong → If ✗, max 2 points)
- [ ] **Image-Answer Consistency**: Does the chosen option match what the completed image shows? (✓ = matches / ✗ = contradicts → If ✗, give 1 point)
- [ ] **Mentions Image**: Explicitly references the completed image? (✓ = multiple times / ~ = once / ✗ = no mention)
- [ ] **Describes Completion**: Describes what the completed area looks like? (✓ = specific / ~ = vague / ✗ = none)
- [ ] **Compares Options**: Compares completion with options A/B/C/D? (✓ = multiple / ~ = 1-2 / ✗ = none)

**Scoring Formula**: 
- Answer WRONG (✗) → max 2 points (failed reasoning, didn't truly use image)
- Answer contradicts image → 1 point
- Answer CORRECT + 4✓ alignment → 5 points
- Answer CORRECT + 3✓+1~ → 4 points
- Answer CORRECT + 2-3✓ → 3 points

### For Multi-View Reasoning Tasks:
- [ ] **Answer Correctness (CHECK FIRST)**: Does rotation direction match Ground Truth? (✓ = correct / ✗ = wrong → If ✗, max 2 points)
- [ ] **Image-Answer Consistency**: Does rotation match what the wider view indicates? (✓ = matches / ✗ = contradicts → If ✗, give 1 point)
- [ ] **Mentions Image**: Explicitly references the wider view? (✓ = multiple times / ~ = once / ✗ = no mention)
- [ ] **Describes Spatial Layout**: Describes spatial relationships in wider view? (✓ = specific / ~ = general / ✗ = none)
- [ ] **Analyzes Objects**: Analyzes object positions across views? (✓ = multiple objects / ~ = one object / ✗ = none)

**Scoring Formula**: 
- Answer WRONG (✗) → max 2 points (failed reasoning, didn't truly use image)
- Answer contradicts image → 1 point
- Answer CORRECT + 4✓ alignment → 5 points
- Answer CORRECT + 3✓+1~ → 4 points
- Answer CORRECT + 2-3✓ → 3 points

## Important Notes
- **CHECK ANSWER CORRECTNESS FIRST**: If answer is wrong (doesn't match GT), max 2 points - model failed to truly use the image
- **CHECK IMAGE-ANSWER CONSISTENCY**: If answer contradicts what the generated image shows, give 1 point
- Wrong answer = failed reasoning alignment, regardless of how well the text describes the image
- If unsure between scores, prefer lower score
- Use checklist for objective scoring

Return JSON format:
{{
    "reasoning_alignment_score": <1-5>,
    "reasoning": "<Part 1: Answer Correctness (vs GT). Part 2: Image-Answer Consistency. Part 3: Checklist with ✓/✗/~. Final score with justification.>"
}}
"""

