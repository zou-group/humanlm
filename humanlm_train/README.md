### Submodule Setup

```bash
git submodule update --init --recursive
```

### Update the submodule to the latest commit

```bash
cd humanlm_train/verl-recipe-humanlm
git checkout humanlm
git pull origin humanlm
```

To start training, follow the instructions in `verl-recipe-humanlm/humanlm/README.md` on the humanlm branch.
