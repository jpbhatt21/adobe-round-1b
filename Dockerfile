# Specify the base image and platform
FROM --platform=linux/amd64 python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application script and the reqs folder into the container
COPY app.py .
COPY pdf_parser.py .
COPY PDFs /app/PDFs

# Create input and output directories
# These directories will be used for mounting volumes during runtime
RUN mkdir -p /app/input /app/output

# Command to run the Python script when the container starts
CMD ["python", "app.py"]