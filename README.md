# A Thousand Self-Organizing LLMs

A cellular automata system where each cell is a differently prompted LLM that can only communicate with its neighbors. This creates an emergent self-organizing system where solutions to complex problems emerge through local interactions.

## Overview

This project implements a novel approach to problem-solving using a grid of LLM agents arranged in a cellular automata topology. Each cell:

- Has its own system prompt with slight variations
- Can only communicate with immediate neighbors (up, down, left, right)
- Receives context from neighbor messages and its own system prompt
- Generates solutions that are evaluated and shared with neighbors
- Learns and improves through iterative communication

## Architecture

### Cellular Automata Topology

The system uses a **k × k grid** (configurable) where:
- Each cell is an independent LLM agent
- Communication is restricted to von Neumann neighborhood (4 neighbors)
- Information propagates through the grid via neighbor-to-neighbor messages
- Each iteration allows cells to update based on neighbor feedback

### LLM Integration

- **Model**: Gemma 3 1B IT (smallest Gemma model)
- **Framework**: JAX for fast parallel inference
- **Inference**: Each cell uses the Gemma model with its unique system prompt
- **Context**: Combines system prompt + neighbor messages + conversation history

### Task: Circle Packing

As a first test, the system solves the **circle packing problem**:
- Given a unit square, pack as many circles as possible
- All circles have the same radius
- No overlaps allowed
- Circles must be completely inside the square

Each cell generates a solution, which is:
1. **Evaluated** for validity (no overlaps, within bounds)
2. **Scored** based on number of circles and efficiency
3. **Shared** with neighbors in the next iteration
4. **Visualized** to show performance across the grid

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- `jax` and `jaxlib` for fast inference
- `gemma` for the LLM model
- `numpy` for numerical operations
- `matplotlib` for visualization

## Usage

### Basic Usage

Run with default settings (5×5 grid, 3 iterations):

```bash
python main.py
```

### Custom Configuration

```bash
python main.py \
    --grid_size 10 \
    --iterations 5 \
    --task circle_packing \
    --output_dir outputs \
    --visualize
```

### Arguments

- `--grid_size`: Size of the k × k grid (default: 5)
- `--iterations`: Number of update iterations (default: 3)
- `--task`: Task to solve (default: circle_packing)
- `--output_dir`: Directory for outputs (default: outputs)
- `--visualize`: Show visualizations during execution

## Output

The system generates:

1. **Score Heatmaps**: Visual representation of cell performance across the grid
2. **Best Solutions**: Visualizations of top-k solutions
3. **Solutions File**: Text file with all cell solutions and scores
4. **Statistics**: Average, maximum, and minimum scores per iteration

## How It Works

1. **Initialization**: 
   - Create k × k grid of cells
   - Each cell gets a unique system prompt with task description
   - Initialize Gemma model once (shared across cells)

2. **Iteration**:
   - For each cell:
     - Collect messages from neighbors
     - Construct prompt: system prompt + neighbor messages + history
     - Generate solution using Gemma
     - Evaluate solution and compute score
     - Update cell state

3. **Communication**:
   - Cells share their solutions with neighbors
   - Neighbor messages provide context for improvement
   - Information propagates through the grid over iterations

4. **Evaluation**:
   - Solutions are parsed and validated
   - Scores computed based on task-specific metrics
   - Best solutions identified and visualized

## Key Features

- **Configurable Grid Size**: Scale from small (3×3) to large (20×20+) grids
- **Parallel Inference**: Uses JAX for efficient computation
- **Neighbor Communication**: Restricted communication creates emergent behavior
- **Task-Agnostic**: Framework can be extended to other tasks
- **Visualization**: Real-time performance tracking and solution visualization

## Extending to Other Tasks

To add a new task:

1. Update `_get_task_description()` in `cellular_llm.py` with task description
2. Implement evaluation function in `_get_evaluator()`
3. Add visualization if needed in `visualize.py`

## Future Improvements

- [ ] True parallel updates using JAX vmap
- [ ] Support for diagonal neighbors (Moore neighborhood)
- [ ] Dynamic grid topologies
- [ ] Multi-turn conversations per cell
- [ ] Adaptive system prompts based on performance
- [ ] Distributed execution across multiple devices

## Citation

If you use this code, please cite:

```
A Thousand Self-Organizing LLMs: Cellular Automata with LLM Cells
```

## License

[Add your license here]

