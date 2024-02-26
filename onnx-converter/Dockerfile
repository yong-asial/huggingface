# Use the official Python image from the Docker Hub
FROM python:3.10.12

# Copy the requirements.txt file into our work directory
COPY requirements.txt .

RUN pip install --upgrade pip

# Install any packages in requirements.txt
RUN pip install -r requirements.txt

# Set the working directory to /apps
WORKDIR /apps

# Copy the current directory contents into the container at /apps
COPY . /apps

# Set the command to run when the container starts
CMD ["bash"]