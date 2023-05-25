# Introduction

## a brief test to compare models

instruments = ["IH.CFE", "IF.CFE", "IH.CFE", "IM.CFE"]

for every 30 minutes in a trading day, a model is constructed

```python
tids = [
    "T01",  # 10:00 
    "T02",  # 10:30 
    "T03",  # 11:00 
    "T04",  # 13:00 
    "T05",  # 13:30 
    "T06",  # 14:00 
    "T07",  # 14:30 
]
```

trailing_windows = [6, 12, 24] months

models = [

+ linear regression
+ mlpr
+ mlpc

]

to summary, a total of (4+1)*(7+1)*3*3=360 models are construct at the end of each month

tested time window 201801-202305

cost_rate = 5e-4

## conclusion

for mlpr and mlpc, parameters for mlp is fixed with  

```python
{
    "hidden_layer_sizes": (5, 5),
    "solver": "adam",
    "random_state": 0,
    "alpha": 1.0,
    "max_iter": 2000,
    "learning_rate": "constant",
    "learning_rate_init": 0.001,
}

```

+ mlpc seems promising, sharpe ratio = 0.24 with  ("IC.CFE", "T02", 12)
+ mlpr is shit
+ lm is just soso


