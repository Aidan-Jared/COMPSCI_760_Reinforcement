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

