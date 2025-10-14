# COMPSCI 760 - Reinforcement Learning Projects

This repository contains projects related to reinforcement learning for the COMPSCI 760 course. It includes an agent for playing Atari games.

## Structure

-   `Atari/`: Contains the implementation of a Feudal Network agent for playing Atari games like Ms. Pac-Man.
---

## Atari Project

This project implements a Feudal Network, a hierarchical reinforcement learning model, to play Atari games using the Gymnasium library.
Comparing it to other models.

### Setup

These instructions are generally applicable to macOS, Linux, and Windows Subsystem for Linux (WSL).

1.  **Clone the repository:**
    ```sh
    git clone <repository-url>
    cd COMPSCI_760_Reinforcement
    ```

2.  **Create a Python virtual environment:**
    It is recommended to use a virtual environment to manage dependencies.
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```
    On Windows, it is recommended to use WSL (Windows Subsystem for Linux). You can find installation instructions on the [official Microsoft documentation](https://learn.microsoft.com/en-us/windows/wsl/install). If not using WSL, you can run `.\venv\Scripts\activate` in Command Prompt or PowerShell.

3.  **Install dependencies:**
    You will need to install PyTorch, Gymnasium, and other related packages.

    -   **For GPU Acceleration:**
        -   **NVIDIA:** Ensure you have the latest NVIDIA drivers and the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) installed.
        -   **AMD:** Ensure you have the latest drivers and the [ROCm platform](https://rocm.docs.amd.com/en/latest/deploy/install.html) installed (Linux only).
        -   Visit the [PyTorch website](https://pytorch.org/get-started/locally/) to get the correct `pip` command for your specific CUDA or ROCm version.

    -   **For CPU or Apple Silicon:**
        If you don't have a compatible GPU or are on an Apple Silicon Mac, you can install the standard PyTorch package.
        ```sh
        pip install torch torchvision torchaudio
        ```
        For Apple Silicon (macOS), PyTorch will automatically use the 'mps' backend for acceleration.

    -   **Install other packages:**
        ```sh
        pip install gymnasium[atari] ale-py
        pip install matplotlib numpy
        ```

### Running the Atari Agent

You can train the agent using the `main.py` script in the `Atari` directory.

-   **To train the Feudal Network model on Ms. Pac-Man:**
    ```sh
    python Atari/main.py --model feudal --env-name "ALE/MsPacman-v5" --run-name "feudal_pacman"
    ```

-   **To train the Transformer-based model:**
    ```sh
    python Atari/main_t.py --env-name "ALE/MsPacman-v5" --run-name "transformer_pacman"
    ```

-   **To test a trained model:**
    ```sh
    python Atari/test.py --model "Atari/models/MsPacman-v5_feudalv4_seed=0_step=3200.pt"
    ```

Training progress and logs will be saved in the `Atari/logs/` directory, and models will be saved in `Atari/models/`.

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

## Contributions:
Aidan Jared was the group leader and worked on the API for _Balatro_, implementing the feudal network, transformer model, and training loop for both feudal and transformer models, as well as reward wrappers for _Ms. Pac-Man_ and _Montezuma's Revenge_, testing loop, all utility code, and player data collection script.Dominicus worked on evaluating model performance using _rliable_, worked on the reward wrapper for _Montezuma's Revenge_, and Documentation. Thomas designed the Q-model and its associated training and testing loops.
