# Use the official Python image.
FROM python:3.10.14-slim

# Set the working directory in the container.
WORKDIR /app

# Copy the requirements file into the container.
COPY requirements.txt ./requirements.txt

# Install the dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container.
COPY . /app

# Expose port 8501 for Streamlit.
EXPOSE 8501

# Run the Streamlit app.
CMD ["streamlit", "run", "streamlit_measurements.py", "--server.port=8501", "--server.address=0.0.0.0"]
