#!/usr/bin/env python3
"""
A Thousand Self-Organizing LLMs: Research Idea Cellular Automata

Each cell is a differently prompted LLM that can only "see" neighbors' ideas.
The system evolves over time as cells propose new research ideas, trying to
balance:
  - Diversity (being different from neighbors)
  - Feasibility and concreteness
  - Building on good neighbor ideas

Usage:
    python research_ca.py \
        --grid_size 50 \
        --iterations 50 \
        --update_fraction 0.05 \
        --visualize

This will also create a GIF visualizing idea flow over iterations.
"""

import os
import re
import json
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import imageio.v2 as imageio

# Try to import gemma
try:
    from gemma import gm
except ImportError:
    print("Error: The 'gemma' library is not installed.")
    print("Please install it using: pip install gemma")
    exit(1)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CellState:
    """State of a single cell in the grid."""
    position: Tuple[int, int]
    system_prompt: str
    conversation_history: List[Dict[str, str]]
    idea: Optional[str] = None
    score: float = 0.0              # heuristic "quality/diversity" score
    last_updated: int = -1          # iteration index when last updated
    sampler: Optional[object] = None  # ChatSampler instance for this cell


# ---------------------------------------------------------------------------
# Core grid
# ---------------------------------------------------------------------------

class CellularIdeaGrid:
    """Grid of LLM cells collaboratively generating research ideas."""
    
    def __init__(
        self,
        grid_size: int = 50,
        task_description: str = "research_ideas",
        update_fraction: float = 0.05,
    ):
        """
        Initialize the cellular automata grid.
        
        Args:
            grid_size: Size of the k x k grid.
            task_description: Description of the task to solve.
            update_fraction: Fraction of cells to update each iteration (0–1).
        """
        self.grid_size = grid_size
        self.task_description = task_description
        self.update_fraction = max(0.0, min(1.0, update_fraction))

        print("Loading Gemma model...")
        self.model = gm.nn.Gemma3_1B()
        self.params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_1B_IT)

        print("Initializing grid...")
        self.grid = self._initialize_grid()

        print("Initializing samplers for each cell...")
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell = self.grid[i, j]
                cell.sampler = gm.text.ChatSampler(
                    model=self.model,
                    params=self.params,
                    multi_turn=True,
                )

    # ----------------------------------------------------------------------
    # Initialization and prompts
    # ----------------------------------------------------------------------

    def _initialize_grid(self) -> np.ndarray:
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

    def _get_task_description(self) -> str:
        if self.task_description == "research_ideas":
            return (
                "You are part of a large grid of researchers. "
                "Each cell proposes short but concrete research ideas. "
                "You see what your immediate neighbors have proposed and you "
                "try to generate a new idea that is:\n"
                "- Feasible and well-scoped\n"
                "- Different enough from your neighbors to expand the search space\n"
                "- Able to build on especially strong neighbor ideas"
            )
        else:
            return self.task_description

    def _generate_system_prompt(self, i: int, j: int) -> str:
        """Generate a unique system prompt for each cell to encourage diversity."""
        base_prompt = f"""You are a research scientist in a huge virtual lab.

TASK:
{self._get_task_description()}

You control cell at position ({i}, {j}) in a 2D grid. You can see only the ideas
of your immediate neighbors (up, down, left, right) and some of your own history.

Your job is to write *one* concrete research idea each time you are asked.

General rules:
- Aim to propose ideas that could realistically be executed by a small team (1–5 people).
- Be specific (what to build, how to evaluate, what data, what metrics).
- Either explore a new direction OR extend/improve a particularly strong neighbor idea.
"""

        # A small set of specialization biases to diversify the grid
        variations = [
            "Focus on AI / ML methodology and benchmarks.",
            "Focus on human-computer interaction and UX aspects.",
            "Focus on cognitive science / psychology related angles.",
            "Focus on societal impact, safety, and ethics.",
            "Focus on tools, infrastructure, and evaluation frameworks.",
            "Focus on multimodal systems (text + images/audio/video).",
            "Focus on theory, formal analysis, and guarantees.",
            "Focus on real-world applications in science or engineering.",
        ]
        variation = variations[(i * self.grid_size + j) % len(variations)]

        return base_prompt + "\nSpecialization: " + variation

    # ----------------------------------------------------------------------
    # Neighborhood utilities
    # ----------------------------------------------------------------------

    def _get_neighbors(self, i: int, j: int) -> List[Tuple[int, int]]:
        neighbors = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                neighbors.append((ni, nj))
        return neighbors

    def _get_neighbor_ideas(self, i: int, j: int) -> str:
        neighbors = self._get_neighbors(i, j)
        if not neighbors:
            return "No neighbors."

        messages = []
        for ni, nj in neighbors:
            cell = self.grid[ni, nj]
            if cell.idea:
                messages.append(
                    f"Neighbor ({ni}, {nj}) [score={cell.score:.2f}, "
                    f"last_updated={cell.last_updated}]:\n{cell.idea[:400]}"
                )
        if not messages:
            return "Neighbors have no ideas yet."
        return "\n\n".join(messages)

    def _format_history(self, history: List[Dict[str, str]], max_len: int = 600) -> str:
        if not history:
            return "No previous ideas."
        # use last assistant message
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                return msg.get("content", "")[:max_len]
        return "No previous ideas from you."

    # ----------------------------------------------------------------------
    # Idea scoring (heuristic, purely local)
    # ----------------------------------------------------------------------

    def _score_idea(self, i: int, j: int) -> float:
        """
        Heuristic scoring of a cell's idea based on:
        - Length & concreteness
        - Diversity vs neighbors (Jaccard overlap on content words)
        - Synergy with strong neighbors (rewards building on high-score neighbors)
        """
        cell = self.grid[i, j]
        if not cell.idea:
            return 0.0

        text = cell.idea.lower()
        words = re.findall(r"\w+", text)
        content_words = [w for w in words if len(w) > 3]
        word_set = set(content_words)

        # Length / concreteness score: prefer mid-length ideas
        n = len(content_words)
        if n == 0:
            length_score = 0.0
        else:
            # Rough triangular: best around 40–80 content words
            center = 60.0
            spread = 40.0
            length_score = max(0.0, 1.0 - abs(n - center) / spread)
            length_score = min(1.0, length_score)

        # Diversity vs neighbors: Jaccard distance
        neighbors = self._get_neighbors(i, j)
        overlaps = []
        for ni, nj in neighbors:
            neigh = self.grid[ni, nj]
            if not neigh.idea:
                continue
            n_words = re.findall(r"\w+", neigh.idea.lower())
            n_content = [w for w in n_words if len(w) > 3]
            n_set = set(n_content)
            if not n_set:
                continue
            inter = len(word_set & n_set)
            union = len(word_set | n_set)
            jaccard = inter / union if union > 0 else 0.0
            overlaps.append(jaccard)

        if overlaps:
            avg_j = sum(overlaps) / len(overlaps)
            diversity_score = 1.0 - avg_j  # more overlap -> lower diversity
        else:
            diversity_score = 1.0  # no neighbor ideas yet => count as diverse

        # Synergy: reward if neighbors already have high scores
        best_neighbor_score = max(
            (self.grid[ni, nj].score for ni, nj in neighbors),
            default=0.0
        )
        synergy = min(0.3, best_neighbor_score / 10.0)  # small boost

        final_score = 0.5 * length_score + 0.4 * diversity_score + 0.1 * synergy
        return float(max(0.0, min(1.5, final_score)))  # clamp a bit

    # ----------------------------------------------------------------------
    # Single cell update
    # ----------------------------------------------------------------------

    def step(self, cell_i: int, cell_j: int, iteration: int) -> str:
        """
        Perform one update step for a single cell (asynchronous update).

        Args:
            cell_i: Row index
            cell_j: Column index
            iteration: Current global iteration (for last_updated)
        """
        cell = self.grid[cell_i, cell_j]

        neighbor_ideas = self._get_neighbor_ideas(cell_i, cell_j)
        history_text = self._format_history(cell.conversation_history)

        prompt = f"""{cell.system_prompt}

---
NEIGHBOR IDEAS (from previous iterations):
{neighbor_ideas}

---
YOUR PREVIOUS IDEA:
{history_text}

---
INSTRUCTION:

Based on your neighbors and your own history, propose ONE new, concrete, feasible
research idea. Try to:

- Be different from your neighbors while still being realistic.
- If a neighbor has an especially strong idea, you may extend or refine it.
- Be specific about: problem, method, data, and evaluation.

FORMAT (STRICT):
[CATEGORY] Short title
1–3 sentences describing the idea.

Examples of categories: [ML], [HCI], [Safety], [Evaluation], [NLP], [Vision].

Do NOT add any extra commentary, greetings, or markdown beyond the format above.
"""

        try:
            response = cell.sampler.chat(prompt)
            if hasattr(response, "text"):
                idea_text = response.text
            elif isinstance(response, str):
                idea_text = response
            else:
                idea_text = str(response)

            idea_text = idea_text.strip()
            cell.idea = idea_text

            # Update score and history
            cell.score = self._score_idea(cell_i, cell_j)
            cell.last_updated = iteration

            cell.conversation_history.append({
                "role": "user",
                "content": "Generate another improved or novel research idea."
            })
            cell.conversation_history.append({
                "role": "assistant",
                "content": idea_text
            })

            return idea_text

        except Exception as e:
            print(f"Error generating idea for cell ({cell_i}, {cell_j}): {e}")
            return ""

    # ----------------------------------------------------------------------
    # Iteration loop
    # ----------------------------------------------------------------------

    def run_iteration(self, iteration: int):
        """
        Run one asynchronous iteration where only a fraction of cells update.
        """
        total_cells = self.grid_size * self.grid_size
        k = max(1, int(self.update_fraction * total_cells))
        print(
            f"\n--- Iteration {iteration} --- "
            f"(updating {k}/{total_cells} cells, "
            f"{self.update_fraction*100:.1f}% of grid)"
        )

        # Choose a random subset of cells to update
        all_indices = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        chosen_flat = np.random.choice(len(all_indices), size=k, replace=False)

        for idx in chosen_flat:
            i, j = all_indices[idx]
            self.step(i, j, iteration)

        # Simple stats
        scores = self.get_scores_grid()
        print(f"  Max score: {np.max(scores):.3f}")
        print(f"  Avg score: {np.mean(scores):.3f}")
        print(f"  Min score: {np.min(scores):.3f}")

    # ----------------------------------------------------------------------
    # Accessors
    # ----------------------------------------------------------------------

    def get_scores_grid(self) -> np.ndarray:
        scores = np.zeros((self.grid_size, self.grid_size), dtype=float)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                scores[i, j] = self.grid[i, j].score
        return scores

    def get_last_updated_grid(self) -> np.ndarray:
        lu = np.full((self.grid_size, self.grid_size), -1, dtype=int)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                lu[i, j] = self.grid[i, j].last_updated
        return lu

    def get_ideas_grid(self) -> np.ndarray:
        ideas = np.empty((self.grid_size, self.grid_size), dtype=object)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                ideas[i, j] = self.grid[i, j].idea
        return ideas

    # ----------------------------------------------------------------------
    # Saving
    # ----------------------------------------------------------------------

    def save_outputs(self, output_dir: str, iteration: int):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        scores = self.get_scores_grid()
        ideas = self.get_ideas_grid()
        lu = self.get_last_updated_grid()

        data = {
            "scores": scores.tolist(),
            "last_updated": lu.tolist(),
            "ideas": [[ideas[i, j] for j in range(self.grid_size)]
                      for i in range(self.grid_size)],
        }

        outfile = os.path.join(output_dir, f"grid_iter_{iteration}.json")
        with open(outfile, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved grid state for iteration {iteration} -> {outfile}")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_grid(
    scores: np.ndarray,
    iteration: int,
    output_dir: str,
    last_updated: Optional[np.ndarray] = None,
):
    """
    Visualize the grid scores as a heatmap.

    - Color encodes heuristic score (quality/diversity).
    - If last_updated is provided, we annotate a few cells.
    """
    if scores.size == 0:
        print("No scores to visualize.")
        return

    plt.figure(figsize=(8, 6))

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "idea_flow",
        ["#FF6B6B", "#FFF2C0", "#6BFF6B"]
    )

    min_val = np.min(scores)
    max_val = np.max(scores)
    if min_val == max_val:
        min_val = max(0.0, min_val - 0.1)
        max_val = max_val + 0.1
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)

    plt.imshow(scores, cmap=cmap, norm=norm, interpolation="nearest")
    plt.colorbar(label="Idea score (quality/diversity)")

    plt.title(f"Research Idea CA - Iteration {iteration}")
    plt.xlabel("X")
    plt.ylabel("Y")

    # Optionally annotate a few cells with last_updated
    if last_updated is not None and scores.shape[0] <= 20 and scores.shape[1] <= 20:
        for i in range(scores.shape[0]):
            for j in range(scores.shape[1]):
                lu = last_updated[i, j]
                text = f"{scores[i, j]:.2f}\n({lu})"
                plt.text(j, i, text, ha="center", va="center", fontsize=6, color="black")

    img_file = os.path.join(output_dir, f"idea_grid_iter_{iteration}.png")
    plt.tight_layout()
    plt.savefig(img_file, dpi=150)
    plt.close()
    print(f"Saved heatmap -> {img_file}")


def make_gif_from_frames(
    output_dir: str,
    pattern_prefix: str = "idea_grid_iter_",
    gif_name: str = "idea_flow.gif",
    duration: float = 0.5,
):
    """Create a GIF from saved PNG frames."""
    frames = []
    files = sorted(
        f for f in os.listdir(output_dir)
        if f.startswith(pattern_prefix) and f.endswith(".png")
    )

    if not files:
        print("No frames found to build GIF.")
        return

    for fname in files:
        path = os.path.join(output_dir, fname)
        frames.append(imageio.imread(path))

    gif_path = os.path.join(output_dir, gif_name)
    imageio.mimsave(gif_path, frames, duration=duration)
    print(f"Saved GIF animation -> {gif_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Research Idea Cellular Automata with LLM cells"
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=10,
        help="Size of the k x k grid (default: 50)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=30,
        help="Number of iterations (default: 30)",
    )
    parser.add_argument(
        "--update_fraction",
        type=float,
        default=0.1,
        help="Fraction of cells to update each iteration (0–1, default: 0.05)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="research_ideas",
        help="Task description tag (default: research_ideas)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="idea_outputs",
        help="Directory to store outputs (default: idea_outputs)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="If set, generate heatmaps and a GIF",
    )

    args = parser.parse_args()

    np.random.seed(42)

    print("Initializing Research Idea Grid...")
    grid = CellularIdeaGrid(
        grid_size=args.grid_size,
        task_description=args.task,
        update_fraction=args.update_fraction,
    )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Initial state (iteration 0)
    grid.save_outputs(args.output_dir, 0)
    if args.visualize:
        scores0 = grid.get_scores_grid()
        lu0 = grid.get_last_updated_grid()
        visualize_grid(scores0, 0, args.output_dir, lu0)

    # Run iterations
    for it in range(1, args.iterations + 1):
        grid.run_iteration(it)
        grid.save_outputs(args.output_dir, it)

        if args.visualize:
            scores = grid.get_scores_grid()
            lu = grid.get_last_updated_grid()
            visualize_grid(scores, it, args.output_dir, lu)

    if args.visualize:
        make_gif_from_frames(args.output_dir)

    print("Simulation complete.")


if __name__ == "__main__":
    main()
