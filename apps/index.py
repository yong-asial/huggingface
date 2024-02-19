import os
import sys
from transformers import pipeline

def load_model_from_local(task_name, model_name):
  pt_save_directory = "/apps/model/pipeline-local-" + model_name
  classifier = pipeline(task_name, model=pt_save_directory)
  return classifier

def get_pipeline(task_name, model_name):
  pt_save_directory = "/apps/model/pipeline-local-" + model_name

  # check if the model available locally
  if os.path.exists(pt_save_directory):
    print("load model from local")
    classifier=load_model_from_local(task_name, model_name)
  else:
    print("load model from huggingFace")
    classifier = pipeline(task_name, model=model_name)
    # save model
    print("save model for next time usage")
    classifier.model.save_pretrained(pt_save_directory)
    classifier.tokenizer.save_pretrained(pt_save_directory)

  # return classifier
  return classifier

def predict_sentiment(task_name, model_name, sentence):
    classifier = get_pipeline(task_name, model_name)
    result = classifier(sentence)
    return result

def main():
    if len(sys.argv) > 3:
        task_name = sys.argv[1]
        model_name = sys.argv[2]
        sentence = sys.argv[3]
        print("Input task name: ", task_name)
        print("Input model name: ", model_name)
        print("Input sentence: ", sentence)
        print(predict_sentiment(task_name, model_name, sentence))
    else:
        print("Please provide a model name and a sentence as command line arguments.")
        print('python3 index.py "task_name" "model_name" "sentence"')


if __name__ == "__main__":
    main()

# tested model names
# "text-classification" "nlptown/bert-base-multilingual-uncased-sentiment" "this movie is so great, I will watch it again and again!"
# "text-classification" "lxyuan/distilbert-base-multilingual-cased-sentiments-student" "私はこの映画が大好きなので、何度でも見ます！"
# "translation_XX_to_YY" "Helsinki-NLP/opus-mt-ja-en" "これはとてもクールです。"

# ['audio-classification', 'automatic-speech-recognition', 'conversational', 
# 'depth-estimation', 'document-question-answering', 'feature-extraction', 'fill-mask',
# 'image-classification', 'image-segmentation', 'image-to-image', 'image-to-text', 'mask-generation',
# 'ner', 'object-detection', 'question-answering', 'sentiment-analysis', 'summarization', 'table-question-answering',
# 'text-classification', 'text-generation', 'text-to-audio', 'text-to-speech', 'text2text-generation',
# 'token-classification', 'translation', 'video-classification', 'visual-question-answering', 'vqa',
# 'zero-shot-audio-classification', 'zero-shot-classification', 'zero-shot-image-classification',
# 'zero-shot-object-detection', 'translation_XX_to_YY']