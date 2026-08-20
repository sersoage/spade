# SPADE Slime Backend Integration


## Overview

SPADE uses Slime as a distributed training backend with SGLang HTTP inference. The integration does not modify Slime's core code; it uses Slime's extension points.

## Key Components

### 1. Custom Data Source (`spade/slime/data_source.py`)

**Purpose**: Bypass Slime's requirement for pre-existing training data files.

SPADE generates games on the fly; `SpadeDataSource` implements Slime's `DataSource` interface without requiring data files.

```python
class SpadeDataSource(DataSource):
    def get_samples(self, num_samples: int) -> List[List[Sample]]:
        return []  # Rollout function generates actual data

    def add_samples(self, samples: List[List[Sample]]):
        pass  # Samples handled by rollout function

    def save(self, rollout_id):
        pass  # No state to save

    def load(self, rollout_id=None):
        pass  # No state to load
```

**Usage in shell script**:
```bash
ROLLOUT_ARGS=(
   --data-source-path spade.slime.data_source.SpadeDataSource
   --rollout-function-path spade.slime.spade_rollout.spade_generate_rollout
   ...
)
```

### 2. Custom Rollout Function (`spade/slime/spade_rollout.py`)

**Purpose**: Implement SPADE's dual-role training loop.

The `spade_generate_rollout()` function:
1. Creates a `SpadeOrchestrator` with the Slime model adapter
2. Calls `collect_trajectories(mode="async")` to generate games and play them
3. Converts `Trajectory` objects to Slime's `Sample` format
4. Returns `RolloutFnTrainOutput(samples, metrics)`

**Key points**:
- Uses `async` mode for concurrent HTTP inference
- Generates games concurrently using `generate_and_save_games_async()`
- Plays games concurrently using `play_games_async()`
- Handles both environment and actor trajectories

### 3. Model Adapter (`spade/slime/model_adapter.py`)

**Purpose**: Adapt SGLang HTTP API to SPADE's `ModelAdapter` interface.

`SlimeModelAdapter` wraps SGLang's HTTP client:
- `generate()`: Synchronous HTTP call
- `generate_async()`: Native async HTTP call
- `generate_batch()`: Batches synchronous HTTP calls
- `apply_template()`: Uses tokenizer's chat template

### 4. Trajectory Converter (`spade/slime/trajectory_converter.py`)

**Purpose**: Convert SPADE `Trajectory` to Slime `Sample`.

`trajectory_to_slime_sample()` maps:
- `observation` → `Sample.prompt`
- `prompt_token_ids + response_token_ids` → `Sample.tokens`
- `reward` → `Sample.reward`
- `metadata` (role, turn, skill, etc.) → `Sample.metadata`

### 5. Arguments (`spade/slime/arguments.py`)

**Purpose**: Add SPADE-specific command-line arguments.

`add_spade_arguments(parser)` adds all SPADE args (returns `parser`):
- Learning potential: `--spade-gamma1`, `--spade-gamma2`
- Environment: `--spade-env-temperature`, `--spade-env-generation-template`
- Actor: `--spade-actor-temperature`, `--spade-actor-template`
- Game generation: `--spade-games-dir`, `--spade-num-games-per-rollout`

### 6. Training Wrapper (`train_spade_slime.py`)

**Purpose**: Entry point that properly initializes SPADE arguments.

```python
def main():
    from slime.utils.arguments import parse_args
    from spade.slime.arguments import add_spade_arguments

    args = parse_args(add_custom_arguments=add_spade_arguments)
    train(args)
```

## Data Flow

```
1. Slime calls spade_generate_rollout(args, rollout_id)
   ↓
2. Create SlimeModelAdapter (SGLang HTTP client)
   ↓
3. Create SpadeOrchestrator(config, model, learning_potential)
   ↓
4. orchestrator.collect_trajectories(mode="async"):
   a. generate_and_save_games_async() → List[Path], Dict[Path, Trajectory]
   b. play_games_async() → List[List[Trajectory]]
   c. Process returns and compute environment rewards
   ↓
5. Convert Trajectories to Slime Samples:
   - Environment trajectories → Samples with role="environment"
   - Actor trajectories → Samples with role="actor"
   ↓
6. Return RolloutFnTrainOutput(samples, metrics)
```

## Logging

Example log output:

```
[GEN-ASYNC] Generating 32 games concurrently
[GEN-ASYNC] Generated 32/32 games
[ASYNC] Statistics: 32 games, avg 12.5 turns (min=5, max=20), avg final reward 0.750, success rate 75.0%
[COLLECT] Final: 400 actor steps, 32 env trajectories
[SPADE] Rollout 0 complete: 400 actor, 32 env samples
```

## Configuration

See `cmd/games/train_spade_4b.sh` for a released configuration example.

Key parameters:
- `--rollout-batch-size`: Number of games per rollout (matches `--spade-num-games-per-rollout`)
- `--spade-game-regeneration-interval`: Regenerate games every N rollouts
- `--spade-games-dir`: Directory for generated game files
- `--spade-gamma1`, `--spade-gamma2`: Learning potential parameters

## Non-Invasive Integration


1. **No modifications to Slime core**: All SPADE code lives in `spade/slime/` (outside `slime/slime/`)
2. **Use extension points**:
   - `--data-source-path` for custom data sources
   - `--rollout-function-path` for custom rollout functions
   - `add_custom_arguments` for custom CLI arguments
3. **Wrapper scripts** instead of modifying Slime's entry points

This allows updating Slime independently without merge conflicts.
