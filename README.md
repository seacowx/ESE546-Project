# ESE546-Project

## Report
[Don't Cheat! Catching Spurious Correlation in NLI Tasks](Spurious_Correlation_in_NLI.pdf)

## Project Grade
100/100 (Best Project Award)


Install dependencies from `requirements.txt`

---

## Training

` python train.py --dataset XXX --n_epochs XXX --content XXX`

Choose dataset (snli or hans) via `--dataset`

Choose whether use [all] (premise + hypothesis), [premise] (premise only), or [hypothesis] (hypothesis) via `--content`

Use `--help` flag to checkout other args

---

## Evaluation

` python eval.py --eval_snli X --state_dict XXX --test_size xx --after_augment False`

Save state dict in root directory and choose saved state dict via `--state_dict` (only need to specify name of the state dict. e.g. to get example.pt, use `--state_dict example`)

Choose whether to evaluate on snli or hans using `--eval_snli`. `1` for eval with snli and `0` for eval using hans.

If the model is trained with augmented data, use `--eval_snli True`

## Data Augmentation with adversarial samples

` python train.py --dataset XXX --n_epochs XXX --content XXX --apply_augment xx --train_size xx --val_size xx`

Choose whether to apply augmentation to training data via `--apply_augment` (default=False).
