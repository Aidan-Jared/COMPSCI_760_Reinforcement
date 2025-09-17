# Dockerfile
FROM rocm/pytorch:rocm6.4.2_ubuntu24.04_py3.12_pytorch_release_2.6.0

WORKDIR /workspace

# (Optional) system deps you need (ffmpeg, git, etc.)
# RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Python deps (add your own)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your project
COPY . .

# Default entrypoint: run your CLI module (can be overridden at `docker run`)
ENTRYPOINT ["python", "-m", "Atari.main"]