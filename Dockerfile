# Use the official Python image from the Docker Hub
FROM python:3.9

# Set the working directory in the container
WORKDIR /code

# Copy the requirements file into the container
COPY ./requirements.txt /code/requirements.txt

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install the dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Set environment variable to allow local file creation (for Grad-CAM etc)
ENV HOME=/tmp

# Expose the port that the app runs on
EXPOSE 7860

# Command to run the application
CMD ["python", "app.py"]
