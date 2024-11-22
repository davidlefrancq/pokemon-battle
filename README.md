# Pokemon Battle

## Requirement
PDM: https://pdm-project.org/en/latest/#installation

## Start client
```bash
pdm run start
```

## Regenerate model
### Prepapre data
```bash
pdm run prepare
```

### Start data visualization
```bash
pdm run visualize
```

### Seach hyperparams
```bash
pdm run optimize
```
Result :
```bash
Temps d execution: 0:08:58.811729
Meilleurs paramétres trouvés:
  bootstrap: False
  max_depth: None
  max_features: sqrt
  min_samples_leaf: 1
  min_samples_split: 5
  n_estimators: 300
Meilleur score: 0.9514
```

### Train model with best hypermarams
```bash
pdm run train
```

### Check execute precdict
```bash
pdm run predict
```