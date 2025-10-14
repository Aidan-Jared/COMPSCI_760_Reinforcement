# COMPSCI_760_Reinforcement

Main code is in the Atari folder. Balatro is legacy code that still works and can be used in the future.

# Using Code
To train a model run `main.py`

### Args
    `lr` learning rate
    `env-name` atari env to train on (*Ms. Pac-Man* fully implimented)
    `num-workers` env to concurently train on
    `num-steps` number of actions between updates
    `max-steps` how many steps to train the model on
    `cuda` allow cuda
    `grad-clip` value for grad norm clipping to stabilize training
    `frame-stacking` should the frames be stacked to give temporal information
    `rgb` train on rgb frames, set to False if frame stacking is True
    `entropy-coef` how much entropy should matter in Feudal loss calculation
    `mlp` use ram instead of images for training
    `time-horizon` Manager horizon (c)
    `dilation` Dilation parameter for manager LSTM
    `hidden-dim-manager` Hidden dim (d) for manager and Q-model
    `hidden-dim-worker` Hidden dim for worker (k)
    `gamma-w` discount factor worker
    `gamma-m` discount factor manager
    `gamma` discount factor for Q
    `alpha` Intrinsic reward coefficient in [0, 1]
    `eps` Random Gausian goal for exploration
    `decay` how much eps decays by
    `decay-limit` how much eps decays to
    `model` model to train
    `seed` random seed
    `run-name` name for logger
    `target-update` steps between target syncs
    `eps-decay-freq` steps between epsilon decays
    `load-file` file to load to continue  training


to test a model run `test.py`