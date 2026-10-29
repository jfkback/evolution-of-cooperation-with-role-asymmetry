"""Run simulation described by a YAML configuration."""

import multiprocessing as mp
import time
from importlib import resources
from pathlib import Path

import fire
import msgspec

from .main_evolution_of_cooperation import one_run


class SimulationConfig(msgspec.Struct, forbid_unknown_fields=True):
    output_dir: str = "Trials"
    group_sizes: list[int] = msgspec.field(
        default_factory=lambda: list(range(2, 11))
    )
    etas: list[float | None] | None = None
    num_trials: int = 1
    num_runs: int = 10000
    N: int = 1000
    m: int = 1000
    beta: float = 1.0
    sigma: float = 0.05
    mu: float = 0.01
    tao: float | None = None

    def __post_init__(self) -> None:
        if self.etas is None:
            self.etas = [None]

        if (
            self.N < 2
            or self.m < 1
            or self.num_trials < 1
            or self.num_runs < 1
        ):
            raise ValueError(
                "N must be at least 2; m, num_trials and num_runs must be positive"
            )

        if not self.group_sizes or any(n < 2 for n in self.group_sizes):
            raise ValueError("group_sizes must contain integers of at least 2")

        if not self.etas or len(set(self.etas)) != len(self.etas):
            raise ValueError("etas must be nonempty and contain no duplicates")

        if any(eta is not None and eta <= 0 for eta in self.etas):
            raise ValueError("eta values must be positive or None")

        if self.beta < 0:
            raise ValueError("beta must be nonnegative")

        if self.sigma < 0:
            raise ValueError("sigma must be nonnegative")

        if not 0 <= self.mu <= 1:
            raise ValueError("mu must be between 0 and 1")

        if self.tao is not None and (not 0 <= self.tao <= 1):
            raise ValueError("tao must be null or between 0 and 1")


def load_config(path: str | Path) -> SimulationConfig:
    """Parse YAML safely and validate its types and parameter ranges."""
    data = Path(path).expanduser().read_bytes()
    return msgspec.yaml.decode(data, type=SimulationConfig)


def run_config(config_path: str | Path) -> None:
    """Run each group-size/eta combination and save each trial's CSV files."""
    path = Path(config_path).expanduser().resolve()
    config = load_config(path)

    run_simulation(config, base_dir=path.parent)


def run_preset(name: str) -> None:
    """Run a packaged configuration, resolving output from the working directory."""
    if name not in ("standard_payoff_config", "probabilistic_payoff_config"):
        raise ValueError(f"Unknown preset: {name}")

    config_data = (
        resources.files("evolution_of_cooperation")
        .joinpath("configs", f"{name}.yaml")
        .read_bytes()
    )
    config = msgspec.yaml.decode(config_data, type=SimulationConfig)
    run_simulation(config, base_dir=Path.cwd())


def run_simulation(config: SimulationConfig, base_dir: Path) -> None:
    """Run a validated configuration, resolving relative output paths from base_dir."""
    output_dir = Path(config.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir

    started = time.monotonic()

    for eta in config.etas:
        for n in config.group_sizes:
            prefix = f"eta_{eta}_" if eta is not None else ""
            suffix = f"n_{n}_{config.num_trials}_{config.N}_{config.m}_{config.beta:g}_{config.sigma:g}_{config.mu:g}"

            if config.tao is not None:
                suffix += f"_tao_{config.tao:g}"

            trial_dir = output_dir / f"{prefix}{suffix}"

            (trial_dir / "Agent Type").mkdir(parents=True, exist_ok=True)
            (trial_dir / "Payoff").mkdir(parents=True, exist_ok=True)

            print(f"Starting simulation for eta={eta}, n={n}", flush=True)
            processes = [
                mp.Process(
                    target=one_run,
                    args=(
                        config.num_runs,
                        i,
                        str(trial_dir),
                        config.N,
                        n,
                        config.m,
                        config.beta,
                        config.sigma,
                        config.mu,
                        eta,
                        config.tao,
                    ),
                )
                for i in range(config.num_trials)
            ]

            for process in processes:
                process.start()

            for process in processes:
                process.join()

            failures = [
                process.exitcode
                for process in processes
                if process.exitcode != 0
            ]

            if failures != []:
                raise RuntimeError(
                    f"Trial processes failed for eta={eta}, n={n}: {failures}"
                )

    print(f"Finished in {time.monotonic() - started:.2f} seconds")


def run(config: str | None = None, preset: str | None = None) -> None:
    """Run a YAML config or packaged preset. Supply exactly one of config or preset."""
    if (config is None) == (preset is None):
        raise ValueError("Provide exactly one of --config or --preset")

    if config is not None:
        run_config(config)
    else:
        run_preset(preset)


def main() -> None:
    """Run the command-line interface."""
    fire.Fire(run)


if __name__ == "__main__":
    main()
