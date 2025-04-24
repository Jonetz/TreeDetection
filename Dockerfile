# Use an official Python base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the setup files first (better for caching)
COPY setup.py .

# Install your package
RUN pip install --no-cache-dir .

# Copy the rest of your project
COPY . .

# Run your app (adjust this based on your entry point)
CMD ["python", "example/example.py"]