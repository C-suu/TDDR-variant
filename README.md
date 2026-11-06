# TDDR variants

Code for **Mitigating Estimation Bias with Representation Learning in TD Error-Driven Regularization**.

| File | Method |
| --- | --- |
| `DADC.py`, `DASC.py`, `SASC.py` | Three variants without representation learning |
| `DADC_R.py`, `DASC_R.py`, `SASC_R.py` | Three representation-enhanced variants |
| `buffer.py` | Replay buffer |
| `main.py` | Training and evaluation |
| `run.sh` | Paper coefficients and seeds 1-5 for five MuJoCo tasks |

The algorithm files use consistent class names and formatting while retaining the existing target constructions and update rules.


## Run

```bash
python main.py --policy DADC-R --env Ant-v2 --seed 1 --qweight 0.6
python main.py --policy DASC --env HalfCheetah-v2 --seed 1 --qweight 0.0
```

The six policy names are `DADC`, `DASC`, `SASC`, `DADC-R`, `DASC-R` and `SASC-R`. `--qweight` is the coefficient upsilon. Training defaults to one million steps, with evaluation every 5,000 steps over ten episodes.

The coefficients are written directly in `run.sh`:

```bash
DRY_RUN=1 bash run.sh  # Preview commands.
bash run.sh            # Train six methods on five tasks, seeds 1-5.
```


