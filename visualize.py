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
    score: Optional[float] = None,
):
    """
    Visualize a circle packing solution with variable radii.
    
    Args:
        solution: String containing the solution
        title: Title for the plot
        save_path: Path to save the figure (optional)
        show: Whether to display the plot
        score: Optional score to display in title
    """
    import re
    
    circles = []
    
    # Try to parse format 1: [(x1, y1, r1), (x2, y2, r2), ...]
    tuple_pattern = r'\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)'
    tuple_matches = re.findall(tuple_pattern, solution)
    
    if tuple_matches:
        # Format 1: (x, y, r) tuples
        for match in tuple_matches:
            try:
                x, y, r = float(match[0]), float(match[1]), float(match[2])
                if r > 0 and 0 <= x <= 1 and 0 <= y <= 1:
                    circles.append((x, y, r))
            except ValueError:
                continue
    else:
        # Try to parse format 2: Circle N: (x, y) radius: r
        coords_pattern = r'\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)'
        coord_matches = list(re.finditer(coords_pattern, solution))
        
        for coord_match in coord_matches:
            try:
                x = float(coord_match.group(1))
                y = float(coord_match.group(2))
                
                # Look for radius in the text after this coordinate
                start_pos = coord_match.end()
                radius_text = solution[start_pos:start_pos + 50]
                
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
                
                if radius is None:
                    global_radius_match = re.search(r'radius[:\s=]+\s*([0-9.]+)', solution.lower())
                    if global_radius_match:
                        radius = float(global_radius_match.group(1))
                
                if radius is not None and radius > 0 and 0 <= x <= 1 and 0 <= y <= 1:
                    circles.append((x, y, radius))
            except (ValueError, IndexError):
                continue
    
    if not circles:
        print("No valid circles found in solution")
        return
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw square boundary
    square = patches.Rectangle((0, 0), 1, 1, linewidth=3, 
                              edgecolor='black', facecolor='white', zorder=0)
    ax.add_patch(square)
    
    # Draw circles with variable radii
    colors = plt.cm.tab20(np.linspace(0, 1, len(circles)))
    sum_radii = 0.0
    
    for i, (x, y, r) in enumerate(circles):
        circle = patches.Circle((x, y), r, linewidth=1.5,
                               edgecolor='darkblue', facecolor=colors[i], 
                               alpha=0.7, zorder=1)
        ax.add_patch(circle)
        # Draw center point
        ax.plot(x, y, 'ko', markersize=4, zorder=2)
        # Label with circle number
        ax.text(x, y, str(i+1), ha='center', va='center', 
               fontsize=7, fontweight='bold', color='white', zorder=3)
        sum_radii += r
    
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    
    # Build title
    title_text = f"{title}\n{len(circles)} circles"
    if score is not None:
        title_text += f", Score (sum of radii): {score:.4f}"
    title_text += f"\nSum of radii: {sum_radii:.4f}"
    
    ax.set_title(title_text, fontsize=12, fontweight='bold')
    ax.set_xlabel('X', fontsize=11)
    ax.set_ylabel('Y', fontsize=11)
    ax.grid(True, alpha=0.2, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved circle packing visualization to {save_path}")
    
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
        title = f"Best Solution #{idx+1} (Cell ({i}, {j}))"
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
            score=score,
        )


def visualize_best_solution(
    grid,
    iteration: int,
    save_dir: Optional[str] = None,
    show: bool = True,
):
    """
    Visualize the single best solution from the grid.
    
    Args:
        grid: CellularLLMGrid instance
        iteration: Current iteration number
        save_dir: Directory to save visualization (optional)
        show: Whether to display plot
    """
    # Find the best solution
    best_score = -1
    best_cell = None
    best_pos = None
    
    for i in range(grid.grid_size):
        for j in range(grid.grid_size):
            cell = grid.grid[i, j]
            if cell.solution and cell.score > best_score:
                best_score = cell.score
                best_cell = cell
                best_pos = (i, j)
    
    if best_cell is None:
        print("No valid solution found to visualize")
        return
    
    title = f"Best Circle Packing Solution - Iteration {iteration}\nCell ({best_pos[0]}, {best_pos[1]})"
    save_path = None
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"best_solution_iter_{iteration}.png")
    
    visualize_circle_packing_solution(
        best_cell.solution,
        title=title,
        save_path=save_path,
        show=show,
        score=best_score,
    )

