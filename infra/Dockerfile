# Use official PyTorch image with GPU support
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y git wget

# Set working directory
WORKDIR /app

# Copy only necessary files
COPY ./src /app/src
COPY ./requirements.txt /app/requirements.txt

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Install your project in an editable mode
RUN pip install -e .

# Expose Streamlit's default port for later use
EXPOSE 8501

# Default command
CMD ["bash"]
