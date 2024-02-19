import os
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# global variables
classifier=None

def load_model_from_local(model_name):
  pt_save_directory = "/apps/model/local-" + model_name
  tokenizer = AutoTokenizer.from_pretrained(pt_save_directory)
  model = AutoModelForSequenceClassification.from_pretrained(pt_save_directory)
  return tokenizer, model

def get_pipeline(task_name, model_name):
  pt_save_directory = "/apps/model/local-" + model_name
  model=None
  tokenizer=None

  # check if the model available locally
  if os.path.exists(pt_save_directory):
    print("load model from local")
    tokenizer, model = load_model_from_local(model_name)
  else:
    print("load model from huggingFace")
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # save model locally (for usage next time)
    tokenizer.save_pretrained(pt_save_directory)
    model.save_pretrained(pt_save_directory)

  # return classifier
  classifier = pipeline(task_name, model=model, tokenizer=tokenizer)
  return classifier

def predict_sentiment(sentence):
    global classifier # use global variable
    model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    task_name = "sentiment-analysis"
    if classifier is None:
      print("initialize classifier")
      classifier = get_pipeline(task_name, model_name)
    result = classifier(sentence)
    return result

def main():
    if len(sys.argv) > 1:
        sentence = sys.argv[1]
        print("Input sentence: ", sentence)
        print(predict_sentiment(sentence))
    else:
        print("Please provide a sentence as a command line argument.")
        print('python3 torch.py "私はこの映画が大好きなので、何度でも見ます！"')


if __name__ == "__main__":
    main()

# usage
# python3 torch.py "私はこの映画が大好きなので、何度でも見ます！"