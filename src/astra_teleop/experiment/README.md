## Usage

```bash
cd src
python -m astra_teleop.experiment.collect
python -m astra_teleop.experiment.optimize
python -m astra_teleop.experiment.plot
python -m astra_teleop.experiment.plot_final
```

## Install pytorch3d

```bash
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install ninja
pip install -e .
# see: [text](https://github.com/facebookresearch/pytorch3d/issues/949)
```