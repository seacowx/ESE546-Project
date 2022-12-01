# ESE546-Project
Install dependencies from `requirements.txt`

---

## Training

` python train.py --dataset XXX --n_epochs XXX --content XXX`

Choose dataset (snail or hans) via `--dataset`

Choose whether use [all] (premise + hypothesis), [premise] (premise only), or [hypothesis] (hypothesis) via `--content`

Use `--help` flag to checkout other args

---

## Evaluation

` python eval.py --eval_snli X --state_dict XXX `

Save state dict in root directory and choose saved state dict via `--state_dict` (only need to specify name of the state dict. e.g. to get example.pt, use `--state_dict example`)

Choose whether to evaluate on snli or hans using `--eval_snli`. `1` for eval with snli and `0` for eval using hans.
