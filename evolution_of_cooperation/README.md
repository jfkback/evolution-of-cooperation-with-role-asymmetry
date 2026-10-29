# Evolution of cooperation

```sh
uv sync
# Standard payoffs
uv run evolution-of-cooperation --preset standard_payoff_config
# Probabilistic payoffs
uv run evolution-of-cooperation --preset probabilistic_payoff_config
# Custom configuration
uv run evolution-of-cooperation --config example.yaml
```

Preset results are saved to `Trials/` in the current directory. Edit `example.yaml` to customize parameters and `output_dir`; relative output paths are resolved from the YAML file's directory.
