#!/usr/bin/env python3

"""
A Thousand Self-Organizing LLMs: Cellular Automata with LLM Cells

This version implements a JAX-native parallel generation step
using the 'transformers' library with its Flax (JAX) backend.
There is no sequential fallback.

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

# Try to import transformers, provide specific error if it fails
try:
    from transformers import AutoTokenizer, FlaxGemmaForCausalLM
except ImportError:
    print("Error: The 'transformers' and 'flax' libraries are not installed.")
    print("Please install them using: pip install transformers flax optax")
    exit(1)


@dataclass
class CellState:
    """State of a single cell in the grid"""
    position: Tuple[int, int]
    system_prompt: str
    conversation_history: List[Dict[str, str]]
    solution: Optional[str] = None
    score: float = 0.0
    # The 'sampler' object has been removed, as it's not JAX-compatible


class CellularLLMGrid:
    """Grid of LLM cells that communicate with neighbors"""
    
    def __init__(
        self,
        grid_size: int = 10,
        task_description: str = "circle_packing",
        model_id: str = "google/gemma-3-1b-it",
    ):
        """
        Initialize the cellular automata grid.
        
        Args:
            grid_size: Size of the k x k grid
            task_description: Description of the task to solve
            model_id: The Hugging Face model ID for a Flax-compatible Gemma model
        """
        self.grid_size = grid_size
        self.task_description = task_description
        self.model_id = model_id
        
        # --- JAX-Native Model Loading ---
        print(f"Loading JAX-native model: {self.model_id}...")
        
        # 1. Load the model statefully to get the pre-trained weights
        model_stateful = FlaxGemmaForCausalLM.from_pretrained(
            self.model_id, dtype=jnp.bfloat16
        )
        
        # 2. Extract the params (the weights) from the stateful model
        self.params = model_stateful.params
        
        # 3. Now, create the stateless model "scaffolding" (code only)
        self.model = FlaxGemmaForCausalLM(
            config=model_stateful.config, _do_init=False, dtype=jnp.bfloat16
        )
        
        print("Model parameters extracted and stateless model definition created.")
        
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Set padding token for batching
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Handle multiple eos_token_ids - take the first one for generation
        if hasattr(model_stateful.config, 'eos_token_id'):
            if isinstance(model_stateful.config.eos_token_id, list):
                self.eos_token_id = model_stateful.config.eos_token_id[0]
            else:
                self.eos_token_id = model_stateful.config.eos_token_id
        else:
            self.eos_token_id = self.tokenizer.eos_token_id
        
        # --- Grid and State Initialization ---
        self.grid = self._initialize_grid()
        self.evaluate_solution = self._get_evaluator(task_description)
        self._master_key = jax.random.PRNGKey(42)
        
        # --- JIT-Compilation ---
        print("Compiling JAX parallel generation function (this may take a moment)...")
        
        # This function is JIT-compiled. It takes the batched inputs
        # and runs the model's 'generate' method in parallel.
        def _parallel_generate(
            params, 
            input_ids: jax.Array, 
            attention_mask: jax.Array, 
            prng_key: jax.Array,
            eos_token_id: int
        ) -> jax.Array:
            # The 'generate' function handles batched inference natively
            output_sequences = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                params=params,
                do_sample=True,
                prng_key=prng_key,
                max_new_tokens=300, # Max tokens for solution
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                eos_token_id=eos_token_id,
                pad_token_id=eos_token_id, # Use EOS for padding
            ).sequences
            return output_sequences
            
        # Store the compiled function with static_argnames
        self._jit_parallel_generate = jax.jit(_parallel_generate, static_argnames=['eos_token_id'])
        print("JAX function compiled.")

        
    def _initialize_grid(self) -> np.ndarray:
        """Initialize grid with cells at each position"""
        grid = np.empty((self.grid_size, self.grid_size), dtype=object)
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                system_prompt = self._generate_system_prompt(i, j)
                grid[i, j] = CellState(
                    position=(i, j),
                    system_prompt=system_prompt,
                    conversation_history=[],
                )
        return grid
    
    def _generate_system_prompt(self, i: int, j: int) -> str:
        """Generate a unique system prompt for each cell"""
        # REMOVED example text to stop model from copying it
        base_prompt = f"""You are an optimization agent in a cellular automata system solving circle packing.

TASK: {self._get_task_description()}

YOUR POSITION: Cell ({i}, {j}) in a {self.grid_size}x{self.grid_size} grid

CRITICAL OUTPUT FORMAT:
- You MUST output ONLY a Python list of 26 tuples.
- The format must be: [(x, y, r), (x, y, r), ...]
- NO extra text, explanations, or commentary.
- Start with '[' and end with ']'.

You will receive neighbor solutions and must improve upon them."""
        
        variations = [
            "Try larger radii.", "Focus on efficient packing.", "Minimize wasted space."
        ]
        variation = variations[(i * self.grid_size + j) % len(variations)]
        
        return f"{base_prompt}\n\n{variation}"
    
    def _get_task_description(self) -> str:
        """Get the task description"""
        # REMOVED example text to stop model from copying it
        if self.task_description == "circle_packing":
            return """CIRCLE PACKING PROBLEM:
Given a square of side length 1, place exactly 26 circles such that:
- The sum of their radii is maximized
- No circles overlap
- All circles are completely inside the square
- Each circle can have a different radius

Provide your solution as a list of circles with their centers and radii in the format:
[(x1, y1, r1), (x2, y2, r2), ..., (x26, y26, r26)]
"""
        else:
            return self.task_description
    
    def _get_evaluator(self, task: str):
        """Get the evaluation function for the task"""
        if task == "circle_packing":
            return self._evaluate_circle_packing
        else:
            return lambda solution_str: 0.0
    
    def _evaluate_circle_packing(self, solution: str) -> float:
        """Evaluate a circle packing solution."""
        if not solution: return 0.0
        try:
            # Parse circles with format (x, y, radius)
            coords_pattern = r'\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)'
            matches = re.findall(coords_pattern, solution)
            if not matches: return 0.0
            
            circles = []
            for match in matches:
                try:
                    x, y, r = float(match[0]), float(match[1]), float(match[2])
                    if 0 <= x <= 1 and 0 <= y <= 1 and r > 0:
                        circles.append((x, y, r))
                except ValueError:
                    continue
            
            if len(circles) == 0:
                return 0.0
            
            # Penalize if not exactly 26 circles
            num_circles = len(circles)
            circle_count_penalty = 1.0 if num_circles == 26 else 0.5
            
            total_radius_sum = 0.0
            for i, (x1, y1, r1) in enumerate(circles):
                valid = True
                if not (r1 <= x1 <= 1 - r1 and r1 <= y1 <= 1 - r1):
                    valid = False
                
                if valid:
                    for j, (x2, y2, r2) in enumerate(circles):
                        if i != j:
                            dist = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
                            if dist < (r1 + r2) - 1e-6:
                                valid = False
                                break
                
                if valid:
                    total_radius_sum += r1
            
            return total_radius_sum * circle_count_penalty
            
        except Exception as e:
            print(f"Error evaluating solution: {e}\nSolution: {solution[:100]}...")
            return 0.0

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
        if not neighbors: return "No neighbors available."
        
        messages = []
        for ni, nj in neighbors:
            cell = self.grid[ni, nj]
            
            # --- FEEDBACK LOOP FIX ---
            # ONLY show the solution text if it's valid (score > 0).
            # Otherwise, just report the score to avoid feeding garbage.
            if cell.score > 0 and cell.solution:
                messages.append(f"Neighbor ({ni}, {nj}) Score={cell.score:.2f}:\n{cell.solution[:150]}...")
            else:
                # Don't show the garbage solution.
                messages.append(f"Neighbor ({ni}, {nj}) Score: 0.0 (No valid solution yet)")
            # --- END OF FIX ---

        if not messages:
            return "No messages from neighbors yet."
        
        return "\n\n".join(messages)
    
    def _format_history(self, history: List[Dict[str, str]], cell_score: float) -> str:
        """Format conversation history for prompt"""
        if not history: return "No previous attempts."
        
        # --- FEEDBACK LOOP FIX ---
        # Only show the previous solution if it had a positive score
        if cell_score > 0:
            for msg in reversed(history):
                if msg.get('role') == 'assistant':
                    return f"Your last solution (Score {cell_score:.2f}):\n{msg.get('content', '')[:500]}"
        
        # If score is 0, don't show the bad solution
        return "Your previous attempts scored 0."
        # --- END OF FIX ---

    def _prepare_batched_inputs(self) -> Tuple[jax.Array, jax.Array, jax.Array]:
            """
            [PYTHON STEP]
            Gathers all N prompts as strings, then tokenizes them into a
            single batched JAX array.
            Returns: (input_ids, attention_mask, prng_key)
            """
            N = self.grid_size * self.grid_size
            full_prompts = []

            print("Preparing batched prompts...")
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    cell = self.grid[i, j]
                    
                    neighbor_messages = self._get_neighbor_messages(i, j)
                    history_text = self._format_history(cell.conversation_history, cell.score)
                    
                    # Construct the user message for this cell
                    user_message = f"""NEIGHBOR MESSAGES (FROM PREVIOUS ITERATION):
    {neighbor_messages}

    YOUR PREVIOUS ATTEMPTS:
    {history_text}

    INSTRUCTION:
    Provide an improved solution. Output ONLY the list of 26 circles.
    Format: [(x, y, r), (x, y, r), ...]
    Each circle must have: center coordinates (x, y) and radius r, all between 0 and 1.
    No circles should overlap. All circles must be fully inside the unit square.
    Maximize the sum of all radii.

    Start your response with '[' and output nothing else except the list."""
                    
                    # Use the chat template for proper formatting
                    messages = [
                        {"role": "user", "content": f"{cell.system_prompt}\n\n{user_message}"}
                    ]
                    
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    full_prompts.append(formatted_prompt)

            # Tokenize the entire batch of N prompts
            print(f"Tokenizing {N} prompts...")
            inputs = self.tokenizer(
                full_prompts,
                return_tensors="jax",
                padding="max_length", # Pad all to the same length
                truncation=True,
                # --- THIS IS THE FIX ---
                max_length=2048 # Increased from 1024
                # --- END OF FIX ---
            )
            
            # Create a unique RNG key for this generation step
            self._master_key, gen_key = jax.random.split(self._master_key)
            
            return inputs['input_ids'], inputs['attention_mask'], gen_key

    def _run_parallel_generation(
        self, 
        input_ids: jax.Array, 
        attention_mask: jax.Array, 
        prng_key: jax.Array
    ) -> jax.Array:
        """
        [JAX STEP]
        Runs the JIT-compiled parallel generation function on the
        entire batch of tokenized inputs.
        """
        N = input_ids.shape[0]
        print(f"Running parallel JAX generation for {N} cells...")
        
        # This calls the JIT-compiled function we created in __init__
        output_sequences = self._jit_parallel_generate(
            self.params,
            input_ids,
            attention_mask,
            prng_key,
            self.eos_token_id
        )
        print("Parallel generation complete.")
        return output_sequences

    def _update_grid_state(self, output_sequences: jax.Array):
        """
        [PYTHON STEP]
        De-tokenizes the JAX array outputs back into strings,
        evaluates them, and updates the Python grid state.
        
        Args:
            output_sequences: The full sequences including input + generated tokens
        """
        N = output_sequences.shape[0]
        print(f"De-tokenizing and scoring {N} outputs...")
        
        # Decode the FULL sequence, *keeping* special tokens
        full_response_texts = self.tokenizer.batch_decode(
            output_sequences, 
            skip_special_tokens=False
        )
        
        # This is the special string that marks the start of the model's answer
        generation_prompt = "<start_of_turn>model\n"
        
        flat_grid = self.grid.flatten()
        for idx in range(N):
            cell = flat_grid[idx]
            full_text = full_response_texts[idx]
            
            # Find the start of the model's *actual* answer
            start_gen_idx = full_text.rfind(generation_prompt)
            
            if start_gen_idx != -1:
                # Get just the model's part
                response_text = full_text[start_gen_idx + len(generation_prompt):]
            else:
                # Fallback: couldn't find the template
                response_text = full_text

            # Now, strip any special tokens (like <eos>)
            response_text = response_text.replace(self.tokenizer.eos_token, "").strip()

            # Debug: Print first few responses to verify extraction
            if idx < 3:
                print(f"\nCell {idx} raw output (first 200 chars): {response_text[:200]}")
            
            # Extract the solution from the response
            solution_text = ""
            
            # Find the first '[' that starts a list
            start_idx = response_text.find('[')
            if start_idx != -1:
                # Find the matching closing ']'
                end_idx = response_text.rfind(']')
                if end_idx != -1 and end_idx > start_idx:
                    solution_text = response_text[start_idx:end_idx+1]
            
            if not solution_text or solution_text == "[]":
                solution_text = response_text # Store the raw response if parsing fails
                if not solution_text:
                    solution_text = "[]" # Default to empty list
            
            # Debug: Print extracted solution for first few cells
            if idx < 3:
                print(f"Cell {idx} extracted solution (first 200 chars): {solution_text[:200]}")
            
            # Evaluate the extracted solution
            score = self.evaluate_solution(solution_text)
            
            # Update the cell's state
            cell.solution = solution_text
            cell.score = score
            cell.conversation_history.append({
                'role': 'user',
                'content': "Provide an improved solution." # Simplified
            })
            cell.conversation_history.append({
                'role': 'assistant',
                'content': solution_text
            })

    def run_iteration(self, num_steps: int = 1):
        """
        Run iterations where each cell updates in parallel.
        """
        for step in range(num_steps):
            print(f"\n--- Running JAX-Native Iteration {step + 1}/{num_steps} ---")
            
            # 1. Prepare inputs (Python -> JAX)
            input_ids, attention_mask, prng_key = self._prepare_batched_inputs()
            
            # 2. Run generation (JAX)
            output_sequences = self._run_parallel_generation(
                input_ids, 
                attention_mask, 
                prng_key
            )
            
            # 3. Update state (JAX -> Python)
            self._update_grid_state(output_sequences)
            
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
        description="A Thousand Self-Organizing LLMs: Cellular Automata (JAX-Native)"
    )
    parser.add_argument(
        "--grid_size", type=int, default=5, help="Size of the k x k grid (default: 5)"
    )
    parser.add_argument(
        "--iterations", type=int, default=3, help="Number of update iterations (default: 3)"
    )
    parser.add_argument(
        "--task", type=str, default="circle_packing", help="Task to solve (default: circle_packing)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Directory for outputs (default: outputs)"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Show visualizations during execution"
    )
    
    args = parser.parse_args()
    
    print("Initializing JAX-Native Cellular LLM Grid...")
    grid = CellularLLMGrid(
        grid_size=args.grid_size,
        task_description=args.task
    )
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # Initial state (Iteration 0)
    if args.visualize:
        visualize_grid(grid.get_scores_grid(), 0, args.output_dir)
    grid.save_outputs(args.output_dir, 0)
    
    # Run iterations
    for i in range(args.iterations):
        # This one function now runs the full, parallel iteration
        grid.run_iteration(num_steps=1)
        
        if args.visualize:
            visualize_grid(grid.get_scores_grid(), i + 1, args.output_dir)
        grid.save_outputs(args.output_dir, i + 1)

    print("Simulation complete.")


if __name__ == "__main__":
    main()