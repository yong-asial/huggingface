# Sentiment Analysis Application

This application uses a Hugging Face model to perform sentiment analysis on a sentence provided as a command line argument.

## Project Structure

- `apps/`: Contains the Python script for the application.
- `apps/model`: Contains the saved model locally.
- `docker-compose.yml`: Docker Compose configuration file.
- `Dockerfile`: Dockerfile for building the Docker image. It might takes around 9GB.
- `requirements.txt`: Contains Python dependencies for the application.
  - Pytorch: install `torchvision` and `torchaudio` for using Vision and Audio models.
  - Tensorflow: install `tensorflow` for using tensorflow models.

## Setup

### Build

```bash
docker-compose build
```

### Run

```bash
docker-compose up -d
```

## Usage

Run the application with a sentence as a command line argument:

```bash
docker-compose run python-server python3 index.py "私はこの映画が大好きなので、何度でも見ます！"
```

or

```bash
docker exec -it python-server bash
python3 index.py "私はこの映画が大好きなので、何度でも見ます！"
```

## Resources

- [Google Colab](https://colab.research.google.com/drive/1sWXmi8xaBUw6-ZYi3y76ODw50jM1jxJb)