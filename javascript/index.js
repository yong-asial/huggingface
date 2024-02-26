import { pipeline, env } from '@xenova/transformers';

env.localModelPath = './models/';
env.allowRemoteModels = false; // set to false, to use model locally

let pipe;
let out;

// if the model has onnx, it can be used directly (need to change env.allowRemoteModels to true to download model from huggingface)
// pipe = await pipeline('sentiment-analysis', 'Xenova/distilbert-base-uncased-finetuned-sst-2-english');
// out = await pipe('I love transformers!');

// 1. if the model does not have onnx, first convert it to onnx using onnx-converter docker
// python convert.py --quantize --model_id google-bert/bert-base-uncased
// 2. then use the model
// pipe = await pipeline('fill-mask', 'google-bert/bert-base-uncased');
// out = await pipe("Hello I'm a [MASK] model.");

// python convert.py --quantize --model_id nlptown/bert-base-multilingual-uncased-sentiment
pipe = await pipeline('sentiment-analysis', 'nlptown/bert-base-multilingual-uncased-sentiment');
out = await pipe('I love transformers!');

console.log(out);
