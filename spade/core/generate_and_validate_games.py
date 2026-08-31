#!/usr/bin/env python3
"""Generate and validate synthetic games for training/evaluation.

Supports both OpenRouter API and Tinker sampling_client for game generation.
"""

import argparse
import asyncio
import os
import random
import sys
from pathlib import Path
from typing import Any

from tqdm.asyncio import tqdm

# Add project root to path
sys.path.append('.')

from spade.core.proofpack_bridge import proofpack_available, validate_game_with_proofpack
from spade.core.game_generator import SyntheticGameGenerator
from spade.core.envs.synthetic_game_env import make_synthetic_env


def test_game_difficulty(
    game_file: str,
    num_test_runs: int = 20,
    num_guesses_per_run: int = 5,
    max_success_rate: float = 0.15,
    *,
    max_turns: int = 20,
    proofpack_target: Any | None = None,
) -> tuple[bool, float]:
    """
    Test if a game has appropriate difficulty by trying random guesses.

    Args:
        game_file: Path to the game file
        num_test_runs: Number of test episodes to run
        num_guesses_per_run: Number of random guesses per episode
        max_success_rate: Maximum acceptable success rate (e.g., 0.15 = 15%)

    Returns:
        (is_valid, success_rate): Whether difficulty is appropriate and the actual success rate
    """
    successes = 0
    native_env = None
    try:
        if proofpack_target is None:
            native_env = make_synthetic_env(
                game_file,
                max_turns=max_turns,
                respect_game_max_turns=True,
            )
        for run in range(num_test_runs):
            env = proofpack_target.instantiate() if proofpack_target is not None else native_env
            try:
                env.reset(seed=run)
                for _ in range(min(num_guesses_per_run, max_turns)):
                    random_answer = str(random.randint(1, 100))
                    _, reward, terminated, truncated, _ = env.step(
                        f'\\boxed{{{random_answer}}}'
                    )
                    if reward > 0:
                        successes += 1
                        break
                    if terminated or truncated:
                        break
            finally:
                if proofpack_target is not None:
                    close = getattr(env, "close", None)
                    if callable(close):
                        close()
    finally:
        if native_env is not None:
            native_env.close()

    success_rate = successes / num_test_runs
    is_valid = success_rate <= max_success_rate

    return is_valid, success_rate


async def generate_and_validate_game_async(
    generator: SyntheticGameGenerator,
    skill: str,
    skill_idx: int,
    game_file: str,
    difficulty: str = 'medium',
    max_attempts: int = 5,
    test_difficulty: bool = True,
    max_success_rate: float = 0.15,
    use_proofpack_qualification: bool = False,
    proofpack_seeds: list[int] | None = None,
    proofpack_timeout_seconds: float = 5.0,
    max_turns: int = 20,
) -> tuple[str, bool]:
    """
    Generate and validate a single game with retries (async version).

    Args:
        generator: SyntheticGameGenerator instance
        skill: Skill category for the game
        skill_idx: Index of the skill (for display)
        game_file: Path where to save the game
        difficulty: Difficulty level ('easy', 'medium', 'hard')
        max_attempts: Maximum number of generation attempts
        test_difficulty: Whether to test difficulty with random guesses
        max_success_rate: Maximum acceptable random success rate
        use_proofpack_qualification: Require the optional ProofPack V0-V4 gate
        proofpack_seeds: Deterministic seeds used by the ProofPack gate
        proofpack_timeout_seconds: Per-execution ProofPack timeout
        max_turns: Rollout horizon used by ProofPack's multi-turn oracle

    Returns:
        (skill, success): Tuple of skill name and whether generation succeeded
    """
    print(f'\n{"="*70}')
    print(f'[{skill_idx+1}] Generating: {skill}')
    print(f'{"="*70}')

    for attempt in range(max_attempts):
        print(f'\n[{skill_idx+1}] Attempt {attempt+1}/{max_attempts} - Generating {skill} game (difficulty: {difficulty})...')

        try:
            # Run in executor to avoid blocking if using sync OpenRouter
            loop = asyncio.get_event_loop()
            game_spec = await loop.run_in_executor(
                None,
                lambda: generator.generate_game(skill, difficulty=difficulty)
            )

            proofpack_target = None
            if use_proofpack_qualification:
                print(f'[{skill_idx+1}]   Running ProofPack V0-V4 formal qualification ladder...')
                is_valid, reason = validate_game_with_proofpack(
                    game_spec.code,
                    seeds=proofpack_seeds,
                    timeout_seconds=proofpack_timeout_seconds,
                    max_turns=max_turns,
                )
                if not is_valid:
                    print(f'[{skill_idx+1}]   ✗ ProofPack rejected game: {reason}')
                    raise ValueError(f'ProofPack qualification failed: {reason}')
                print(f'[{skill_idx+1}]   ✓ ProofPack qualified: {reason}')
                # Reuse ProofPack's replay-backed proxy for the remaining
                # generation-time probes. Candidate code never executes in this
                # process merely because the assurance gate was enabled.
                from proofpack_env.spade_target import SpadeEnvironmentTarget

                proofpack_target = SpadeEnvironmentTarget(
                    game_spec.code,
                    action_format="boxed",
                    max_turns=max_turns,
                    operation_timeout_seconds=proofpack_timeout_seconds,
                )

            generator.save_game(game_spec, game_file)
            print(f'[{skill_idx+1}]   ✓ Saved to {game_file}')
            print(f'[{skill_idx+1}]   Game name: {game_spec.game_name}')

            print(f'[{skill_idx+1}]   Testing game can be loaded...')
            env = (
                proofpack_target.instantiate()
                if proofpack_target is not None
                else make_synthetic_env(
                    game_file,
                    max_turns=max_turns,
                    respect_game_max_turns=True,
                )
            )
            try:
                obs, info = env.reset()
            finally:
                env.close()
            obs_preview = obs[:100] + '...' if len(obs) > 100 else obs
            print(f'[{skill_idx+1}]   ✓ Reset successful. Initial obs: {obs_preview}')

            if test_difficulty:
                print(f'[{skill_idx+1}]   Testing difficulty level (should have low random success rate)...')
                is_valid, success_rate = test_game_difficulty(
                    game_file,
                    num_test_runs=20,
                    num_guesses_per_run=5,
                    max_success_rate=max_success_rate,
                    max_turns=max_turns,
                    proofpack_target=proofpack_target,
                )

                print(f'[{skill_idx+1}]   Random success rate: {success_rate:.1%} (threshold: {max_success_rate:.1%})')

                if not is_valid:
                    print(f'[{skill_idx+1}]   ✗ Game is too easy! Random success rate {success_rate:.1%} > {max_success_rate:.1%}')
                    raise ValueError(f'Game difficulty too low: {success_rate:.1%} success rate')

            print(f'[{skill_idx+1}]   ✓✓ Game validated successfully!')
            print(f'[{skill_idx+1}] ✅ SUCCESS: {skill} game generated and validated')
            return (skill, True)

        except Exception as e:
            print(f'[{skill_idx+1}]   ✗ Game validation failed: {e}')
            if os.path.exists(game_file):
                os.remove(game_file)
                print(f'[{skill_idx+1}]   Removed faulty game file')

            if attempt == max_attempts - 1:
                print(f'[{skill_idx+1}]   ERROR: Failed to generate valid game after {max_attempts} attempts')
                print(f'[{skill_idx+1}] ❌ FAILED to generate valid game for: {skill}')
                return (skill, False)
            else:
                print(f'[{skill_idx+1}]   Retrying...')

    return (skill, False)


async def main_async():
    parser = argparse.ArgumentParser(
        description='Generate and validate synthetic games for training/evaluation'
    )

    # Required arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save generated games'
    )

    # Generation backend
    parser.add_argument(
        '--use_tinker',
        action='store_true',
        help='Use Tinker sampling client for game generation (default: use OpenRouter from env)'
    )

    # Model configuration
    parser.add_argument(
        '--model',
        type=str,
        default='qwen/qwen3-coder-plus',
        help='Model name for game generation (default: qwen/qwen3-coder-plus)'
    )
    parser.add_argument(
        '--renderer',
        type=str,
        default='qwen3',
        help='Renderer for Tinker (only used with --use_tinker, default: qwen3)'
    )
    parser.add_argument(
        '--tinker_base_url',
        type=str,
        default=None,
        help='Tinker service base URL (optional)'
    )

    # Game configuration
    parser.add_argument(
        '--skills',
        nargs='+',
        default=['Pattern Recognition', 'Mathematical Reasoning'],
        help='Skills to generate games for (default: Pattern Recognition, Mathematical Reasoning)'
    )
    parser.add_argument(
        '--difficulty',
        type=str,
        default='medium',
        choices=['easy', 'medium', 'hard'],
        help='Game difficulty level (default: medium)'
    )
    parser.add_argument(
        '--max_attempts',
        type=int,
        default=5,
        help='Maximum generation attempts per game (default: 5)'
    )

    # Validation configuration
    parser.add_argument(
        '--skip_difficulty_test',
        action='store_true',
        help='Skip random difficulty testing'
    )
    parser.add_argument(
        '--max_success_rate',
        type=float,
        default=0.15,
        help='Maximum acceptable random success rate (default: 0.15 = 15%%)'
    )
    parser.add_argument(
        '--use_proofpack_qualification',
        action='store_true',
        help=(
            'Require ProofPack V0-V4 qualification. Fails closed if a compatible '
            'proofpack_env installation is unavailable.'
        ),
    )
    parser.add_argument(
        '--proofpack_seeds',
        type=int,
        nargs='+',
        default=[0, 1, 42],
        help='Seeds used by ProofPack qualification (default: 0 1 42)',
    )
    parser.add_argument(
        '--proofpack_timeout_seconds',
        type=float,
        default=5.0,
        help='Per-execution ProofPack timeout in seconds (default: 5)',
    )
    parser.add_argument(
        '--max_turns',
        type=int,
        default=20,
        help='Maximum rollout horizon used during ProofPack qualification (default: 20)',
    )

    args = parser.parse_args()

    if args.use_proofpack_qualification:
        available, detail = proofpack_available()
        if not available:
            parser.error(
                "ProofPack qualification was requested, but a compatible "
                f"proofpack_env installation is unavailable: {detail}"
            )
        print(
            "ProofPack isolates qualification replays only; accepted games are "
            "loaded in-process by SPADE's native runtime."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('='*70)
    print('Synthetic Game Generation and Validation')
    print('='*70)
    print()
    print(f'Output directory: {output_dir}')
    print(f'Skills: {args.skills}')
    print(f'Difficulty: {args.difficulty}')
    print(f'Max attempts per game: {args.max_attempts}')
    print()

    # Initialize game generator based on backend
    if args.use_tinker:
        print(f'Using Tinker with model: {args.model}')
        try:
            import tinker
            from tinker_cookbook.renderers import get_renderer
            from tinker_cookbook.tokenizer_utils import get_tokenizer

            service_client = tinker.ServiceClient(base_url=args.tinker_base_url)
            sampling_client = service_client.create_sampling_client(base_model=args.model)

            tokenizer = get_tokenizer(args.model)
            renderer = get_renderer(args.renderer, tokenizer=tokenizer)

            generator = SyntheticGameGenerator(
                model=args.model,
                use_tinker=True,
                tinker_sampling_client=sampling_client,
                tinker_renderer=renderer,
            )
        except ImportError:
            print('ERROR: Tinker not available. Install with: pip install tinker')
            sys.exit(1)
    else:
        print(f'Using OpenRouter API with model: {args.model}')
        generator = SyntheticGameGenerator(
            model=args.model,
        )

    print()
    print(f'Generating {len(args.skills)} games in parallel...')
    print()

    tasks = []
    for skill_idx, skill in enumerate(args.skills):
        game_file = str(output_dir / f'game_{skill_idx:03d}_{skill.lower().replace(" ", "_")}.py')

        task = generate_and_validate_game_async(
            generator=generator,
            skill=skill,
            skill_idx=skill_idx,
            game_file=game_file,
            difficulty=args.difficulty,
            max_attempts=args.max_attempts,
            test_difficulty=not args.skip_difficulty_test,
            max_success_rate=args.max_success_rate,
            use_proofpack_qualification=args.use_proofpack_qualification,
            proofpack_seeds=args.proofpack_seeds,
            proofpack_timeout_seconds=args.proofpack_timeout_seconds,
            max_turns=args.max_turns,
        )
        tasks.append(task)

    results = await tqdm.gather(*tasks, desc='Generating games')

    failed_skills = [skill for skill, success in results if not success]

    print()
    print('='*70)
    print('GENERATION SUMMARY')
    print('='*70)
    print(f'Total skills: {len(args.skills)}')
    print(f'Successful: {len(args.skills) - len(failed_skills)}')
    print(f'Failed: {len(failed_skills)}')

    if failed_skills:
        print('\nFailed skills:')
        for skill in failed_skills:
            print(f'  - {skill}')
        sys.exit(1)
    else:
        print('\n✅ All games generated successfully!')
        print(f'Games saved to: {output_dir}')
        sys.exit(0)


def main():
    """Sync wrapper to run async main."""
    asyncio.run(main_async())


if __name__ == '__main__':
    main()
