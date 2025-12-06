# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install a comprehensive set of system dependencies required by OpenCV and other CV libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Prevent paddleocr from trying to write to a non-existent home directory
ENV PADDLEOCR_HOME=/app/.paddleocr
RUN mkdir -p $PADDLEOCR_HOME

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Railway will use the Procfile to run the application.
# If you were not using Railway's Procfile support, you would use:
# CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "camera_app:app"]
