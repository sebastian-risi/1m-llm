"""Visualization utilities for the cellular LLM grid"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
import matplotlib.patches as patches


def visualize_grid_scores(
    scores: np.ndarray,
    title: str = "Cell Performance Scores",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Visualize cell scores as a heatmap.
    
    Args:
        scores: 2D array of scores (grid_size x grid_size)
        title: Title for the plot
        save_path: Path to save the figure (optional)
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    im = ax.imshow(scores, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Score', rotation=270, labelpad=20)
    
    # Add text annotations with scores
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            score = scores[i, j]
            text_color = 'white' if score < 0.5 else 'black'
            ax.text(j, i, f'{score:.2f}', ha='center', va='center',
                   color=text_color, fontsize=8, fontweight='bold')
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, scores.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, scores.shape[0], 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_circle_packing_solution(
    solution: str,
    title: str = "Circle Packing Solution",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Visualize a circle packing solution.
    
    Args:
        solution: String containing the solution
        title: Title for the plot
        save_path: Path to save the figure (optional)
        show: Whether to display the plot
    """
    import re
    
    # Extract coordinates
    coords_pattern = r'\(([0-9.]+),\s*([0-9.]+)\)'
    matches = re.findall(coords_pattern, solution)
    
    if not matches:
        print("No valid coordinates found in solution")
        return
    
    circles = []
    for match in matches:
        x, y = float(match[0]), float(match[1])
        if 0 <= x <= 1 and 0 <= y <= 1:
            circles.append((x, y))
    
    # Extract radius
    radius_pattern = r'radius[:\s=]+([0-9.]+)'
    radius_match = re.search(radius_pattern, solution.lower())
    if radius_match:
        radius = float(radius_match.group(1))
    else:
        radius = 0.05  # Default
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw square boundary
    square = patches.Rectangle((0, 0), 1, 1, linewidth=2, 
                              edgecolor='black', facecolor='white')
    ax.add_patch(square)
    
    # Draw circles
    colors = plt.cm.tab20(np.linspace(0, 1, len(circles)))
    for i, (x, y) in enumerate(circles):
        circle = patches.Circle((x, y), radius, linewidth=1,
                               edgecolor='blue', facecolor=colors[i], alpha=0.6)
        ax.add_patch(circle)
        ax.plot(x, y, 'ro', markersize=3)  # Center point
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    ax.set_title(f"{title}\n{len(circles)} circles, radius={radius:.3f}", 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_best_solutions(
    grid,
    top_k: int = 5,
    save_dir: Optional[str] = None,
    show: bool = True,
):
    """
    Visualize the top-k best solutions from the grid.
    
    Args:
        grid: CellularLLMGrid instance
        top_k: Number of top solutions to visualize
        save_dir: Directory to save visualizations (optional)
        show: Whether to display plots
    """
    # Get all cells with scores
    cells_with_scores = []
    for i in range(grid.grid_size):
        for j in range(grid.grid_size):
            cell = grid.grid[i, j]
            if cell.solution:
                cells_with_scores.append((cell, cell.score, (i, j)))
    
    # Sort by score
    cells_with_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Visualize top k
    for idx, (cell, score, (i, j)) in enumerate(cells_with_scores[:top_k]):
        title = f"Best Solution #{idx+1} (Cell ({i}, {j}), Score: {score:.3f})"
        save_path = None
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"solution_{idx+1}_cell_{i}_{j}.png")
        
        visualize_circle_packing_solution(
            cell.solution,
            title=title,
            save_path=save_path,
            show=show,
        )

