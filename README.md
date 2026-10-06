# TEK4090 inverted cart pole control simulation



### SETUP
1) First we create environment and install the dependencies
```console
conda env create -f environment.yml
```
2) Activate envirmonamt
```console
conda activate cartPole-env
```
3) Test to see if everything is correct.
```console
python -c "import numpy, scipy, matplotlib, numba, control, skfuzzy, pybullet; print('ok')"

```

4) Run the code
```console
python .\pendulumSim.py --controller lqr --T 10
```

5) Test more functionality
```console
python .\pendulumSim.py --help
```
