#!/usr/bin/env python3

"""
A Thousand Self-Organizing LLMs: Cellular Automata with LLM Cells

Each cell is a differently prompted LLM that can only communicate with neighbors.
The system uses JAX for fast parallel inference.

Usage:
    python main.py --grid_size 10 --iterations 5 --visualize
"""

import jax
import jax.numpy as jnp
import numpy as np
import json
import re
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Import visualization functions
try:
    from visualize import visualize_best_solution
except ImportError:
    # Fallback if visualize module not available
    visualize_best_solution = None

# Try to import gemma, provide specific error if it fails
try:
    from gemma import gm
except ImportError:
    print("Error: The 'gemma' library is not installed.")
    print("Please install it using: pip install gemma")
    exit(1)


@dataclass
class CellState:
    """State of a single cell in the grid"""
    position: Tuple[int, int]
    system_prompt: str
    conversation_history: List[Dict[str, str]]
    solution: Optional[str] = None
    score: float = 0.0
    num_circles: int = 0  # Number of circles generated in the solution
    sampler: Optional[object] = None  # ChatSampler instance for this cell


class CellularLLMGrid:
    """Grid of LLM cells that communicate with neighbors"""
    
    def __init__(
        self,
        grid_size: int = 10,
        task_description: str = "circle_packing",
    ):
        """
        Initialize the cellular automata grid.
        
        Args:
            grid_size: Size of the k x k grid
            task_description: Description of the task to solve
        """
        self.grid_size = grid_size
        self.task_description = task_description
        
        # Initialize Gemma model (shared across all cells)
        print("Loading Gemma model...")
        self.model = gm.nn.Gemma3_1B()
        self.params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_1B_IT)
        
        # Get tokenizer for parallel processing
        print("Loading tokenizer...")
        try:
            self.tokenizer = gm.text.Tokenizer()
        except Exception:
            # Fallback: try to get tokenizer from sampler
            print("Could not load gm.text.Tokenizer, attempting fallback...")
            temp_sampler = gm.text.ChatSampler(
                model=self.model,
                params=self.params,
                multi_turn=False,
            )
            self.tokenizer = getattr(temp_sampler, 'tokenizer', None)
            if self.tokenizer is None:
                raise RuntimeError("Failed to acquire Gemma tokenizer.")
        
        # Initialize grid
        self.grid = self._initialize_grid()
        
        # Create samplers for each cell (for sequential/fallback mode)
        print("Initializing samplers for each cell (sequential mode)...")
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell = self.grid[i, j]
                # Each cell gets its own sampler to maintain separate conversation state
                cell.sampler = gm.text.ChatSampler(
                    model=self.model,
                    params=self.params,
                    multi_turn=True,
                )
        
        # Task-specific evaluation function
        self.evaluate_solution = self._get_evaluator(task_description)
        
        # JIT-compiled parallel iteration function (will be compiled on first use)
        self._jit_batched_iteration = None
        self._master_key = jax.random.PRNGKey(42)
        
    def _initialize_grid(self) -> np.ndarray:
        """Initialize grid with cells at each position"""
        grid = np.empty((self.grid_size, self.grid_size), dtype=object)
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Each cell gets a slightly different system prompt
                system_prompt = self._generate_system_prompt(i, j)
                
                grid[i, j] = CellState(
                    position=(i, j),
                    system_prompt=system_prompt,
                    conversation_history=[],
                )
        
        return grid
    
    def _generate_system_prompt(self, i: int, j: int) -> str:
        """Generate a unique system prompt for each cell"""
        base_prompt = f"""You are part of a cellular automata system solving a problem collaboratively.

TASK: {self._get_task_description()}

INSTRUCTIONS:
- You can communicate with your immediate neighbors (up, down, left, right)
- Share your solutions, insights, or any information you think is relevant
- Learn from your neighbors' messages
- Provide your best solution to the task
- Be concise and clear in your responses

Your goal is to solve the task and help your neighbors improve their solutions through communication. ONLY OUTPUT THE POSITION AND SIZE OF THE CIRCLES IN THE FORMAT [(x1, y1, r1), (x2, y2, r2), ..., (x26, y26, r26)]. DO NOT USE ANY OTHER TEXT OR MARKDOWN FORMATTING. IT HAS TO BE 26 CIRCLES."""
        
        # Add some variation based on position
        variations = [
            "Focus on efficiency and optimal solutions.",
            "Consider creative and innovative approaches.",
            "Prioritize accuracy and correctness.",
            "Think about scalability and generalization.",
            "Balance simplicity and effectiveness.",
        ]
        variation = variations[(i * self.grid_size + j) % len(variations)]
        
        return f"{base_prompt}\n\n{variation}"
    
    def _get_task_description(self) -> str:
        """Get the task description"""
        if self.task_description == "circle_packing":
            return """CIRCLE PACKING OPTIMIZATION PROBLEM:
Place exactly 26 circles within a UNIT square (side length 1) such that: 
- The sum of all circle radii is MAXIMIZED
- No circles overlap. THIS WILL LEAD TO SCORE OF 0.
- IF IT IS LESS THAN 26 CIRCLES, IT WILL BE A SCORE OF 0.
- All circles remain fully contained within the square boundary
- Each circle can have a different radius

This is a constrained optimization challenge combining discrete placement decisions 
with continuous radius optimization.

Provide your solution as a list of circles, each with center (x, y) and radius r: [(x1, y1, r1), (x2, y2, r2), ..., (x26, y26, r26)]
X AND Y ARE BETWEEN 0 AND 1. R IS GREATER THAN 0 AND LESS THAN 0.5.
DO NOT USE ANY OTHER TEXT OR MARKDOWN FORMATTING.

Your score is the sum of radii of all valid circles (must be exactly 26, no overlaps, all in bounds)."""
        else:
            return self.task_description
    
    def _get_evaluator(self, task: str):
        """Get the evaluation function for the task"""
        if task == "circle_packing":
            return self._evaluate_circle_packing
        else:
            # Return a default evaluator
            return lambda solution_str: (0.0, 0)
    
    def _evaluate_circle_packing(self, solution: str) -> Tuple[float, int]:
        """
        Evaluate a circle packing solution.
        Returns a tuple of (score, num_circles) where:
        - score: sum of radii of valid circles (must be exactly 26, no overlaps, all in bounds)
        - num_circles: number of circles parsed from the solution
        """
        if not solution:
            return 0.0, 0
        
        try:
            circles = []
            
            # Try to parse format 1: [(x1, y1, r1), (x2, y2, r2), ...]
            tuple_pattern = r'\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)'
            tuple_matches = re.findall(tuple_pattern, solution)
            
            if tuple_matches:
                # Format 1: (x, y, r) tuples
                for match in tuple_matches:
                    try:
                        x, y, r = float(match[0]), float(match[1]), float(match[2])
                        if r > 0:
                            circles.append((x, y, r))
                    except ValueError:
                        continue

                print(circles)
            else:
                # Try to parse format 2: Circle N: (x, y) radius: r
                # First, extract all coordinate pairs
                coords_pattern = r'\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)'
                coord_matches = list(re.finditer(coords_pattern, solution))
                
                # Try to find radius for each coordinate pair
                # Look for "radius:" or "radius=" near each coordinate
                for coord_match in coord_matches:
                    try:
                        x = float(coord_match.group(1))
                        y = float(coord_match.group(2))
                        
                        # Look for radius in the text after this coordinate
                        start_pos = coord_match.end()
                        radius_text = solution[start_pos:start_pos + 50]  # Look ahead 50 chars
                        
                        # Try to find radius pattern
                        radius_patterns = [
                            r'radius[:\s=]+\s*([0-9.]+)',
                            r'r[:\s=]+\s*([0-9.]+)',
                        ]
                        
                        radius = None
                        for pattern in radius_patterns:
                            radius_match = re.search(pattern, radius_text, re.IGNORECASE)
                            if radius_match:
                                radius = float(radius_match.group(1))
                                break
                        
                        # If no radius found, try to find a global radius
                        if radius is None:
                            global_radius_match = re.search(r'radius[:\s=]+\s*([0-9.]+)', solution.lower())
                            if global_radius_match:
                                radius = float(global_radius_match.group(1))
                        
                        if radius is not None and radius > 0:
                            circles.append((x, y, radius))
                    except (ValueError, IndexError):
                        continue
            
            # Store the number of circles parsed
            num_circles_parsed = len(circles)
            
            # Must have exactly 26 circles
            if len(circles) != 26:
                # Penalize if not exactly 26
                penalty = abs(len(circles) - 26) * 0.1
                if len(circles) == 0:
                    return 0.0, 0
                # Still evaluate what we have, but with penalty
            
            if len(circles) == 0:
                return 0.0, 0

            # Check validity: no overlaps, all within bounds
            valid_circles = []
            sum_radii = 0.0
            
            for i, (x1, y1, r1) in enumerate(circles):
                valid = True
                
                # Check if circle fits in square (fully contained)
                if x1 - r1 < 0 or x1 + r1 > 1:
                    valid = False
                if y1 - r1 < 0 or y1 + r1 > 1:
                    valid = False
                
                # Check for overlaps with other circles
                if valid:
                    for j, (x2, y2, r2) in enumerate(circles):
                        if i != j:
                            dist = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
                            # Circles overlap if distance < sum of radii
                            if dist < (r1 + r2) - 1e-6:  # Small tolerance for floating point
                                valid = False
                                break
                
                if valid:
                    valid_circles.append((x1, y1, r1))
                    sum_radii += r1
            
            # Penalize if not exactly 26 valid circles
            if len(valid_circles) != 26:
                penalty = abs(len(valid_circles) - 26) * 0.1
                sum_radii = max(0.0, sum_radii - penalty)
            
            return float(sum_radii), num_circles_parsed
            
        except Exception as e:
            print(f"Error evaluating solution: {e}\nSolution: {solution[:200]}...")
            return 0.0, 0
    
    def _get_neighbors(self, i: int, j: int) -> List[Tuple[int, int]]:
        """Get neighbor positions (up, down, left, right)"""
        neighbors = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                neighbors.append((ni, nj))
        return neighbors
    
    def _get_neighbor_messages(self, i: int, j: int) -> str:
        """Get messages from neighbors from the *previous* iteration"""
        neighbors = self._get_neighbors(i, j)
        if not neighbors:
            return "No neighbors available."
        
        messages = []
        for ni, nj in neighbors:
            cell = self.grid[ni, nj]
            if cell.score > 0 and cell.solution:
                messages.append(f"Neighbor ({ni}, {nj}) Score={cell.score:.2f} (Circles: {cell.num_circles}/26):\n{cell.solution[:150]}...")
            elif cell.conversation_history:
                # Fallback to last thing said if no solution yet
                last_msg = cell.conversation_history[-1].get('content', '')
                if last_msg:
                    messages.append(f"Neighbor ({ni}, {nj}) says:\n{last_msg[:150]}...")
        
        if not messages:
            return "No messages from neighbors yet."
        
        return "\n\n".join(messages)
    
    def step(self, cell_i: int, cell_j: int) -> str:
        """
        Perform one step for a single cell (SEQUENTIAL operation).
        
        Args:
            cell_i: Row index of the cell
            cell_j: Column index of the cell
        
        Returns:
            The response from the cell
        """
        cell = self.grid[cell_i, cell_j]
        
        # Get neighbor messages
        neighbor_messages = self._get_neighbor_messages(cell_i, cell_j)
        
        # Construct prompt
        prompt = f"""{cell.system_prompt}

---
NEIGHBOR MESSAGES (FROM PREVIOUS ITERATION):
{neighbor_messages}
---
YOUR PREVIOUS ATTEMPTS (Score: {cell.score:.2f}, Circles: {cell.num_circles}/26):
{self._format_history(cell.conversation_history)}
"""
        
        # Generate response using Gemma
        try:
            # Use this cell's sampler
            response = cell.sampler.chat(prompt)
            
            # Extract text from response
            if hasattr(response, 'text'):
                response_text = response.text
            elif isinstance(response, str):
                response_text = response
            else:
                response_text = str(response)
            
            #print(f"Full prompt: {prompt}")
            #print(f"Response text: {response_text}")
            score, num_circles = self.evaluate_solution(response_text)
            #print(f"Score: {score}, Circles: {num_circles}/26")
            #exit()
            # Update cell state
            cell.conversation_history.append({
                'role': 'user',
                'content': "Provide an improved solution." # Simplified history
            })
            cell.conversation_history.append({
                'role': 'assistant',
                'content': response_text
            })
            cell.solution = response_text
            
            # Evaluate solution
            cell.score, cell.num_circles = self.evaluate_solution(response_text)
            
            return response_text
            
        except Exception as e:
            print(f"Error generating response for cell ({cell_i}, {cell_j}): {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def _format_history(self, history: List[Dict[str, str]], max_length: int = 500) -> str:
        """Format conversation history for prompt"""
        if not history:
            return "No previous attempts."
        
        # Show last assistant/model response
        for msg in reversed(history):
            if msg.get('role') == 'assistant':
                return f"Last solution:\n{msg.get('content', '')[:max_length]}"
        
        return "No previous solutions."
    
    def _prepare_batched_state(self) -> Tuple[jax.Array, List[str], List[str], List[str]]:
        """
        Prepare batched state for parallel processing.
        This is the PRE-TOKENIZATION step.
        """
        N = self.grid_size * self.grid_size
        
        # Create RNG keys for each cell
        self._master_key, *rng_keys = jax.random.split(self._master_key, N + 1)
        batched_rng_keys = jnp.array(rng_keys)
        
        # Prepare system prompts and neighbor messages
        sys_prompts = []
        neighbor_messages_list = []
        history_texts = []
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell = self.grid[i, j]
                
                # Construct the full prompt text for each cell
                neighbor_messages = self._get_neighbor_messages(i, j)
                history_text = self._format_history(cell.conversation_history)
                score_text = f"Your current score is {cell.score:.2f} (Circles: {cell.num_circles}/26)."

                full_prompt = f"""{cell.system_prompt}
---
NEIGHBOR MESSAGES (FROM PREVIOUS ITERATION):
{neighbor_messages}
---
YOUR PREVIOUS ATTEMPTS ({score_text}):
{history_text}
---
INSTRUCTION:
Based on your neighbors and your previous attempts, provide an improved solution.
You must provide exactly 26 circles. Your solution format:
[(x1, y1, r1), (x2, y2, r2), ..., (x26, y26, r26)]

Where each circle has center (x, y) and radius r. Maximize the sum of all radii.
Example (showing first 4 circles):
[(0.1, 0.1, 0.05), (0.3, 0.1, 0.05), (0.1, 0.3, 0.05), (0.3, 0.3, 0.05), ...]
"""
                # For JAX-native, we would tokenize this 'full_prompt'
                # and pass the token array.
                # For now, we pass the text for the sequential fallback.
                sys_prompts.append(full_prompt) # Re-using this list for full prompt
                neighbor_messages_list.append(neighbor_messages) # Not used by fallback
                history_texts.append(history_text) # Not used by fallback

        
        # In a true JAX implementation, we would tokenize 'sys_prompts' here
        # and return JAX arrays.
        # e.g., batched_tokens = self.tokenizer.encode(sys_prompts, ...)
        
        return (
            batched_rng_keys,
            sys_prompts, # This list now contains the FULL prompts
            neighbor_messages_list,
            history_texts
        )
    
    def _single_cell_update_jax(
        self,
        params,
        rng_key: jax.Array,
        full_prompt_tokens: jax.Array,
        # ... other JAX-native state ...
    ) -> Tuple[jax.Array, jax.Array]:
        """
        [PLACEHOLDER] Single cell update function for JAX vmap.
        This is the JAX-native version that should work with vmap.
        
        TODO: Implement using a JAX-native generation loop.
        """
        
        # 1. CALL JAX-NATIVE GENERATION FUNCTION
        # This is the missing piece. You need a function that performs
        # autoregressive sampling using model.apply() in a JAX-compatible way
        # (e.g., inside a jax.lax.while_loop).
        
        # output_tokens = jax_gemma_generate(params, rng_key, full_prompt_tokens, self.model)
        
        # Placeholder: return input tokens
        output_tokens = full_prompt_tokens 
        
        # 2. CALL JAX-NATIVE EVALUATION FUNCTION
        # The _evaluate_circle_packing (with regex) is not JAX-compatible.
        # You would need a JAX-native parser and evaluator, which is
        # extremely difficult.
        
        # A more realistic JAX-native approach:
        # - LLM returns tokens
        # - The vmap function *only* returns the output tokens
        # - Evaluation (parsing, scoring) happens *outside* vmap, in Python.
        
        score = jnp.array(0.0)  # Placeholder
        
        return output_tokens, score
    
    def _run_parallel_iteration(self):
        """
        Run a single iteration using JAX vmap (or fallback to sequential).
        """
        N = self.grid_size * self.grid_size
        
        print(f"WARNING: True JAX parallelization (_single_cell_update_jax) is not yet implemented.")
        print(f"The `ChatSampler` is a Python wrapper and cannot be used with `jax.vmap`.")
        print(f"Falling back to sequential processing for {N} cells...")
        
        # Prepare all prompts first
        batched_rng_keys, full_prompts, _, _ = self._prepare_batched_state()
        
        # --- SEQUENTIAL FALLBACK ---
        # This loop uses the pre-initialized cell.sampler
        idx = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell = self.grid[i, j]
                full_prompt = full_prompts[idx]
                
                print(f"Processing cell ({i}, {j})...")
                
                try:
                    response = cell.sampler.chat(full_prompt)
                    
                    if hasattr(response, 'text'):
                        response_text = response.text
                    elif isinstance(response, str):
                        response_text = response
                    else:
                        response_text = str(response)
                    
                    print(f"Full prompt: {full_prompt}")
                    print(f"Response text: {response_text}")
                    score, num_circles = self.evaluate_solution(response_text)
                    print(f"Score: {score}, Circles: {num_circles}/26")
                    # Update cell state
                    exit()
                    cell.solution = response_text
                    cell.score, cell.num_circles = self.evaluate_solution(response_text)
                    cell.conversation_history.append({
                        'role': 'user',
                        'content': "Provide an improved solution." # Simplified
                    })
                    cell.conversation_history.append({
                        'role': 'assistant',
                        'content': response_text
                    })
                    
                except Exception as e:
                    print(f"Error updating cell ({i}, {j}): {e}")
                
                idx += 1
    
    def run_iteration(self, num_steps: int = 1, parallel: bool = False):
        """
        Run iterations where each cell updates.
        
        Args:
            num_steps: Number of update steps
            parallel: Whether to attempt parallel update
        """
        for step in range(num_steps):
            print(f"\n--- Running Iteration {step + 1}/{num_steps} ---")
            
            if parallel:
                # This will currently use the sequential fallback
                self._run_parallel_iteration()
            else:
                # Explicitly sequential processing
                print(f"Running sequential update for {self.grid_size * self.grid_size} cells...")
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        print(f"Processing cell ({i}, {j})...")
                        try:
                            self.step(i, j)
                        except Exception as e:
                            print(f"Error updating cell ({i}, {j}): {e}")
            
            # Print stats for the iteration
            scores = self.get_scores_grid()
            print(f"Iteration {step + 1} Complete.")
            print(f"  Max Score: {np.max(scores):.2f}")
            print(f"  Avg Score: {np.mean(scores):.2f}")
            print(f"  Min Score: {np.min(scores):.2f}")

    def get_scores_grid(self) -> np.ndarray:
        """Get scores for all cells as a grid"""
        scores = np.zeros((self.grid_size, self.grid_size))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                scores[i, j] = self.grid[i, j].score
        return scores
    
    def get_solutions_grid(self) -> np.ndarray:
        """Get solutions for all cells as a grid"""
        solutions = np.empty((self.grid_size, self.grid_size), dtype=object)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                solutions[i, j] = self.grid[i, j].solution
        return solutions

    def save_outputs(self, output_dir: str, iteration: int):
        """Save scores and best solution to file."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        scores = self.get_scores_grid()
        solutions = self.get_solutions_grid()
        
        # Save scores
        scores_file = os.path.join(output_dir, f"scores_iter_{iteration}.json")
        with open(scores_file, 'w') as f:
            json.dump(scores.tolist(), f, indent=2)
            
        # Find and save best solution
        best_idx = np.unravel_index(np.argmax(scores), scores.shape)
        best_score = scores[best_idx]
        best_solution = solutions[best_idx]
        
        best_file = os.path.join(output_dir, f"best_solution_iter_{iteration}.json")
        with open(best_file, 'w') as f:
            json.dump({
                "position": (int(best_idx[0]), int(best_idx[1])),
                "score": best_score,
                "solution": best_solution
            }, f, indent=2)
        
        print(f"Outputs for iteration {iteration} saved to {output_dir}")


def visualize_grid(scores: np.ndarray, iteration: int, output_dir: str):
    """
    Visualize the grid scores as a heatmap using Matplotlib.
    """
    if scores.size == 0:
        print("No scores to visualize.")
        return

    plt.figure(figsize=(10, 8))
    
    # Create a colormap from red (low) to green (high)
    cmap = mcolors.LinearSegmentedColormap.from_list("rg", ["#FF6B6B", "#FFF2C0", "#6BFF6B"])
    
    # Handle case where all scores are the same
    min_val = np.min(scores)
    max_val = np.max(scores)
    if min_val == max_val:
        min_val = max(0, min_val - 1)
        max_val = max_val + 1
        
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
    
    plt.imshow(scores, cmap=cmap, norm=norm, interpolation='nearest')
    
    # Add text annotations for scores
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            plt.text(j, i, f"{scores[i, j]:.2f}",
                     ha="center", va="center", color="black", fontsize=8)
            
    plt.colorbar(label="Solution Score")
    plt.title(f"Cellular LLM Scores - Iteration {iteration}")
    plt.xlabel("Cell X-coordinate")
    plt.ylabel("Cell Y-coordinate")
    
    # Save the figure
    img_file = os.path.join(output_dir, f"heatmap_iter_{iteration}.png")
    plt.savefig(img_file)
    print(f"Heatmap saved to {img_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="A Thousand Self-Organizing LLMs: Cellular Automata"
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=2,
        help="Size of the k x k grid (default: 5)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of update iterations (default: 3)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="circle_packing",
        help="Task to solve (default: circle_packing)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory for outputs (default: outputs)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show visualizations during execution"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Attempt to run with JAX parallel (will fallback to sequential)"
    )
    
    args = parser.parse_args()
    
    print("Initializing Cellular LLM Grid...")
    grid = CellularLLMGrid(
        grid_size=args.grid_size,
        task_description=args.task
    )
    
    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # Initial state (Iteration 0)
    if args.visualize:
        visualize_grid(grid.get_scores_grid(), 0, args.output_dir)
    grid.save_outputs(args.output_dir, 0)
    
    # Run iterations
    for i in range(args.iterations):
        grid.run_iteration(num_steps=1, parallel=args.parallel)
        
        # Visualize and save at each step
        if args.visualize:
            visualize_grid(grid.get_scores_grid(), i + 1, args.output_dir)
            # Visualize best circle packing solution
            if visualize_best_solution is not None:
                visualize_best_solution(grid, i + 1, args.output_dir, show=False)
        grid.save_outputs(args.output_dir, i + 1)

    print("Simulation complete.")
    
    # Final visualization of best solution
    if args.visualize and visualize_best_solution is not None:
        print("\nVisualizing final best solution...")
        visualize_best_solution(grid, args.iterations, args.output_dir, show=True)


if __name__ == "__main__":
    main()