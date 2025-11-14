"""
Main script to run the cellular automata LLM system.
"""

import argparse
import numpy as np
from cellular_llm import CellularLLMGrid
from visualize import visualize_grid_scores, visualize_best_solutions
import os


def main():
    parser = argparse.ArgumentParser(
        description="A Thousand Self-Organizing LLMs: Cellular Automata with LLM Cells"
    )
    parser.add_argument(
        '--grid_size',
        type=int,
        default=5,
        help='Size of the k x k grid (default: 5)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=3,
        help='Number of iterations to run (default: 3)'
    )
    parser.add_argument(
        '--task',
        type=str,
        default='circle_packing',
        help='Task to solve (default: circle_packing)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs',
        help='Directory to save outputs (default: outputs)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Show visualizations'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("A Thousand Self-Organizing LLMs")
    print("=" * 60)
    print(f"Grid size: {args.grid_size}x{args.grid_size}")
    print(f"Task: {args.task}")
    print(f"Iterations: {args.iterations}")
    print("=" * 60)
    
    # Initialize grid
    print("\nInitializing cellular automata grid...")
    grid = CellularLLMGrid(
        grid_size=args.grid_size,
        task_description=args.task,
    )
    
    # Run iterations
    print(f"\nRunning {args.iterations} iterations...")
    for iteration in range(args.iterations):
        print(f"\n--- Iteration {iteration + 1}/{args.iterations} ---")
        grid.run_iteration(num_steps=1)
        
        # Get scores
        scores = grid.get_scores_grid()
        avg_score = np.mean(scores)
        max_score = np.max(scores)
        min_score = np.min(scores)
        
        print(f"\nScore statistics after iteration {iteration + 1}:")
        print(f"  Average: {avg_score:.3f}")
        print(f"  Maximum: {max_score:.3f}")
        print(f"  Minimum: {min_score:.3f}")
        
        # Visualize scores
        if args.visualize:
            visualize_grid_scores(
                scores,
                title=f"Cell Performance Scores - Iteration {iteration + 1}",
                save_path=os.path.join(args.output_dir, f"scores_iter_{iteration + 1}.png"),
                show=True,
            )
    
    # Final statistics
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    final_scores = grid.get_scores_grid()
    print(f"Final average score: {np.mean(final_scores):.3f}")
    print(f"Final maximum score: {np.max(final_scores):.3f}")
    print(f"Best cell position: {np.unravel_index(np.argmax(final_scores), final_scores.shape)}")
    
    # Visualize final state
    if args.visualize:
        visualize_grid_scores(
            final_scores,
            title="Final Cell Performance Scores",
            save_path=os.path.join(args.output_dir, "final_scores.png"),
            show=True,
        )
        
        # Visualize best solutions
        print("\nVisualizing best solutions...")
        visualize_best_solutions(
            grid,
            top_k=5,
            save_dir=os.path.join(args.output_dir, "best_solutions"),
            show=args.visualize,
        )
    
    # Save solutions to file
    solutions_file = os.path.join(args.output_dir, "solutions.txt")
    with open(solutions_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("All Cell Solutions\n")
        f.write("=" * 60 + "\n\n")
        
        for i in range(args.grid_size):
            for j in range(args.grid_size):
                cell = grid.grid[i, j]
                f.write(f"\nCell ({i}, {j}) - Score: {cell.score:.3f}\n")
                f.write("-" * 60 + "\n")
                if cell.solution:
                    f.write(f"{cell.solution}\n")
                else:
                    f.write("No solution generated.\n")
                f.write("\n")
    
    print(f"\nSolutions saved to {solutions_file}")
    print(f"Outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()

