# peg-in-hole

## To run
`$ poetry run ./peg_in_hole/main.py`

## Installation
- Add `VORTEX_INSTALLATION_PATH` to your user env. variable (i.e. `C:\CM Labs\Vortex Studio 2023.8`)
- `poetry install`
- `poetry shell`
- `pip install tensorflow`

**pytorch**
`$ poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`

Note:
- To have `.venv` in projec: `poetry config virtualenvs.in-project true`

## TODO
- [ ] Class that interfaces w/ dll
  - Same as KG2_Robot or another class

- [ ] Send commands
- [ ] Log measures
- [ ] Plot logs

## Resources
- Check: [rl-games](https://github.com/Denys88/rl_games)