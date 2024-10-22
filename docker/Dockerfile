# Use the official PyTorch Docker image with GPU support
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

# Set the working directory
WORKDIR /workspace/IRP

# Install necessary packages
RUN apt-get update && \
    apt-get install -y \
    curl \
    screen
#    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy the requirements file from your src directory and install dependencies
COPY src/requirements.txt .

# Install Python packages required by your application
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the entire source directory into the container
COPY src/ /workspace/IRP/src
COPY data/ /workspace/IRP/data

# Expose a port for Jyputer Notebook
EXPOSE 8888

# Change the working directory to /app/src
WORKDIR /workspace/IRP/src

# Set an entrypoint to run the experiment script from within /app/src
ENTRYPOINT ["python3", "run_experiment_with_validation.py"]
