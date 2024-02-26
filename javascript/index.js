import { pipeline, env } from '@xenova/transformers';

env.localModelPath = './models/';
env.allowRemoteModels = false; // set to false, to use model locally

let pipe;
let out;

// pipe = await pipeline('sentiment-analysis', 'Xenova/distilbert-base-uncased-finetuned-sst-2-english');
// out = await pipe('I love transformers!');

// python convert.py --quantize --model_id google-bert/bert-base-uncased
// pipe = await pipeline('fill-mask', 'local-google-bert/bert-base-uncased');
// out = await pipe("Hello I'm a [MASK] model.");

// python convert.py --quantize --model_id nlptown/bert-base-multilingual-uncased-sentiment
pipe = await pipeline('sentiment-analysis', 'nlptown/bert-base-multilingual-uncased-sentiment');
out = await pipe('I love transformers!');

console.log(out);
