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

### Stop

```bash
docker-compose down
```

## Usage

Run the application with a sentence as command line arguments:

```bash
docker exec -it python-server bash
python3 index.py "task_name" "model_name" "sentence"
```

## Pytorch vs. Tensorflow vs. Default

For a model (for example, nlptown/bert-base-multilingual-uncased-sentiment), there are many model binaries, pytorch, tensorflow, jax, etc. We can choose to use which model binary by using specific Tokenizer and Model class.

### Pytorch

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# load pytorch model from hugginFace
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# save model
save_directory = "./directory"
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

# load model from local
tokenizer = AutoTokenizer.from_pretrained(save_directory)
model = AutoModelForSequenceClassification.from_pretrained(save_directory)

# use model
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
classifier("I like huggingFace.")
```

### Tensorflow

Note: to use following code, you need to install `tensorflow` to this docker.

```python
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# load tensorflow model from hugginFace
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# save tensorflow model
save_directory = "./directory"
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

# load model from local
tokenizer = AutoTokenizer.from_pretrained(save_directory)
tf_model = TFAutoModelForSequenceClassification.from_pretrained(save_directory)

# use model
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
classifier("I like huggingFace.")
```

### Default

Otherwise, we just use `pipeline` function to import default Tokenizer/Model for specified model_name.
We don't need to import Tokenizer and Model class.

```python
from transformers import pipeline

# load default model from huggingFace
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
classifier = pipeline("sentiment-analysis", model=model_name)

# save model
save_directory = "./directory"
classifier.model.save_pretrained(save_directory)
classifier.tokenizer.save_pretrained(save_directory)

# Load the pipeline with the saved model
classifier = pipeline("sentiment-analysis", model=save_directory)
classifier("I like huggingFace.")
```

## Installed Packages

These are required for translation model.

```txt
sentencepiece
sacremoses
```

## Resources

- [Google Colab](https://colab.research.google.com/drive/1sWXmi8xaBUw6-ZYi3y76ODw50jM1jxJb)