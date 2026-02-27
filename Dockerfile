FROM python:3.10-slim

# Install Java (Required for tabula-py)
RUN apt-get update && apt-get install -y \
    default-jre-headless \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with user id 1000
# Hugging Face Spaces requires a non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory
WORKDIR $HOME/app

# Copy requirements and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application code
COPY --chown=user . .

# Hugging Face Spaces expects port 7860
ENV PORT 7860

# Run the application using gunicorn with uvicorn workers for production performance
CMD exec gunicorn --bind :$PORT --workers 1 --worker-class uvicorn.workers.UvicornWorker --threads 8 main:app
