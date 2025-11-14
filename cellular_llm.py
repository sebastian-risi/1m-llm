"""
A Thousand Self-Organizing LLMs: Cellular Automata with LLM Cells

Each cell is a differently prompted LLM that can only communicate with neighbors.
The system uses JAX for fast parallel inference.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import re

try:
    from gemma import gm
except ImportError:
    raise ImportError("Please install gemma: pip install gemma")


@dataclass
class CellState:
    """State of a single cell in the grid"""
    position: Tuple[int, int]
    system_prompt: str
    conversation_history: List[Dict[str, str]]
    solution: Optional[str] = None
    score: float = 0.0
    sampler: Optional[object] = None  # ChatSampler instance for this cell


class CellularLLMGrid:
    """Grid of LLM cells that communicate with neighbors"""
    
    def __init__(
        self,
        grid_size: int = 10,
        task_description: str = "circle_packing",
        use_jit: bool = True,
    ):
        """
        Initialize the cellular automata grid.
        
        Args:
            grid_size: Size of the k x k grid
            task_description: Description of the task to solve
            use_jit: Whether to use JAX JIT compilation
        """
        self.grid_size = grid_size
        self.task_description = task_description
        self.use_jit = use_jit
        
        # Initialize Gemma model (shared across all cells)
        print("Loading Gemma model...")
        self.model = gm.nn.Gemma3_41()
        self.params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_1B_IT)
        
        # Initialize grid (each cell will get its own sampler)
        self.grid = self._initialize_grid()
        
        # Create samplers for each cell
        print("Initializing samplers for each cell...")
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

YOUR POSITION: Cell ({i}, {j}) in a {self.grid_size}x{self.grid_size} grid

INSTRUCTIONS:
- You can communicate with your immediate neighbors (up, down, left, right)
- Share your solutions, insights, or any information you think is relevant
- Learn from your neighbors' messages
- Provide your best solution to the task
- Be concise and clear in your responses

Your goal is to solve the task and help your neighbors improve their solutions through communication."""
        
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
            return """CIRCLE PACKING PROBLEM:
Given a square of side length 1, pack as many circles as possible such that:
- All circles have the same radius
- No circles overlap
- All circles are completely inside the square

Provide your solution as a list of circle centers in the format:
[(x1, y1), (x2, y2), ..., (xn, yn)]

Also specify the radius you used. Try to maximize the number of circles."""
        else:
            return self.task_description
    
    def _get_evaluator(self, task: str):
        """Get the evaluation function for the task"""
        if task == "circle_packing":
            return self._evaluate_circle_packing
        else:
            return lambda x: 0.0
    
    def _evaluate_circle_packing(self, solution: str) -> float:
        """
        Evaluate a circle packing solution.
        Returns a score between 0 and 1.
        """
        if not solution:
            return 0.0
        
        try:
            # Try to extract circle centers and radius from the solution
            # Look for patterns like [(x, y), ...] or (x, y) coordinates
            coords_pattern = r'\(([0-9.]+),\s*([0-9.]+)\)'
            matches = re.findall(coords_pattern, solution)
            
            if not matches:
                return 0.0
            
            # Parse coordinates
            circles = []
            for match in matches:
                x, y = float(match[0]), float(match[1])
                if 0 <= x <= 1 and 0 <= y <= 1:
                    circles.append((x, y))
            
            if len(circles) == 0:
                return 0.0
            
            # Try to extract radius
            radius_pattern = r'radius[:\s=]+([0-9.]+)'
            radius_match = re.search(radius_pattern, solution.lower())
            if radius_match:
                radius = float(radius_match.group(1))
            else:
                # Estimate radius based on number of circles
                # For a rough estimate, assume optimal packing
                radius = min(0.1, 1.0 / (2 * len(circles)**0.5))
            
            # Check for overlaps and validity
            valid_circles = 0
            for i, (x1, y1) in enumerate(circles):
                valid = True
                # Check if circle fits in square
                if x1 - radius < 0 or x1 + radius > 1:
                    valid = False
                if y1 - radius < 0 or y1 + radius > 1:
                    valid = False
                
                # Check for overlaps with other circles
                if valid:
                    for j, (x2, y2) in enumerate(circles):
                        if i != j:
                            dist = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
                            if dist < 2 * radius:
                                valid = False
                                break
                
                if valid:
                    valid_circles += 1
            
            # Score based on number of valid circles and efficiency
            base_score = valid_circles / max(10, len(circles))  # Normalize
            efficiency = valid_circles * (radius ** 2) * np.pi  # Area covered
            score = min(1.0, base_score * 0.7 + efficiency * 0.3)
            
            return score
            
        except Exception as e:
            print(f"Error evaluating solution: {e}")
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
        """Get messages from neighbors"""
        neighbors = self._get_neighbors(i, j)
        if not neighbors:
            return "No neighbors available."
        
        messages = []
        for ni, nj in neighbors:
            cell = self.grid[ni, nj]
            if cell.solution:
                messages.append(f"Neighbor at ({ni}, {nj}) says: {cell.solution[:200]}...")
            elif cell.conversation_history:
                last_msg = cell.conversation_history[-1].get('content', '')
                if last_msg:
                    messages.append(f"Neighbor at ({ni}, {nj}) says: {last_msg[:200]}...")
        
        if not messages:
            return "No messages from neighbors yet."
        
        return "\n".join(messages)
    
    def step(self, cell_i: int, cell_j: int, key: Optional[jax.Array] = None) -> str:
        """
        Perform one step for a single cell.
        
        Args:
            cell_i: Row index of the cell
            cell_j: Column index of the cell
            key: JAX random key (optional)
        
        Returns:
            The response from the cell
        """
        cell = self.grid[cell_i, cell_j]
        
        # Get neighbor messages
        neighbor_messages = self._get_neighbor_messages(cell_i, cell_j)
        
        # Construct prompt
        prompt = f"""{cell.system_prompt}

NEIGHBOR MESSAGES:
{neighbor_messages}

YOUR PREVIOUS ATTEMPTS:
{self._format_history(cell.conversation_history)}

Now provide your solution to the task. Be specific and include all necessary details."""
        
        # Generate response using Gemma
        try:
            # Ensure sampler is initialized
            if cell.sampler is None:
                cell.sampler = gm.text.ChatSampler(
                    model=self.model,
                    params=self.params,
                    multi_turn=True,
                )
            
            # Use this cell's sampler (text-only, no images)
            response = cell.sampler.chat(prompt)
            
            # Extract text from response (handle different response formats)
            if hasattr(response, 'text'):
                response_text = response.text
            elif isinstance(response, str):
                response_text = response
            else:
                # Try to get text attribute or convert to string
                response_text = str(response)
            
            # Update cell state
            cell.conversation_history.append({
                'role': 'user',
                'content': prompt
            })
            cell.conversation_history.append({
                'role': 'assistant',
                'content': response_text
            })
            cell.solution = response_text
            
            # Evaluate solution
            cell.score = self.evaluate_solution(response_text)
            
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
        
        formatted = []
        for msg in history[-3:]:  # Last 3 messages
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')[:max_length]
            formatted.append(f"{role}: {content}")
        
        return "\n".join(formatted)
    
    def run_iteration(self, num_steps: int = 1, parallel: bool = False):
        """
        Run one iteration where each cell updates.
        
        Args:
            num_steps: Number of update steps per cell
            parallel: Whether to update cells in parallel (not fully implemented yet)
        """
        if parallel:
            # For now, sequential updates
            # TODO: Implement true parallel updates with JAX
            pass
        
        for step in range(num_steps):
            print(f"Running iteration step {step + 1}/{num_steps}...")
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    try:
                        self.step(i, j)
                    except Exception as e:
                        print(f"Error updating cell ({i}, {j}): {e}")
    
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

