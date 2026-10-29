# Evolution of cooperation with role asymmetry and the emergence of fairness in the multiplayer dictator game
This is the official codebase for the simulations ran in the paper: _Evolution of cooperation with role asymmetry and the emergence of fairness in the multiplayer dictator game_.

## Install and run with uv

```sh
git clone https://github.com/jfkback/evolution-of-cooperation-with-role-asymmetry.git
cd evolution-of-cooperation-with-role-asymmetry/evolution_of_cooperation

uv sync
# Standard payoffs
uv run evolution-of-cooperation --preset standard_payoff_config
# Probabilistic payoffs
uv run evolution-of-cooperation --preset probabilistic_payoff_config

# Custom configuration
uv run evolution-of-cooperation --config example.yaml
```

Preset results are saved to `Trials/` in the current directory. Edit `example.yaml` to customize parameters and `output_dir`; relative output paths are resolved from the YAML file's directory.

## Install from Git with pip

Requires Python 3.12 or newer and Git.

```sh
python -m pip install "git+https://github.com/jfkback/evolution-of-cooperation-with-role-asymmetry.git#subdirectory=evolution_of_cooperation"
evolution-of-cooperation --preset standard_payoff_config
# Or: evolution-of-cooperation --preset probabilistic_payoff_config
```

For a custom configuration, copy `evolution_of_cooperation/example.yaml` from the repository, edit it, and run `evolution-of-cooperation --config path/to/config.yaml`.

## Reproducing Paper Results
The preset configs only run 10 independent trials. In the paper, we used 1,000 trials for the main results. You can copy the preset configs (`evolution_of_cooperation/src/evolution_of_cooperation/configs/probabilistic_payoff_config.yaml` and `evolution_of_cooperation/src/evolution_of_cooperation/configs/standard_payoff_config.yaml`) and change them to do 1,000 trials to exactly replicate the results.
