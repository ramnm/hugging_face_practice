import tensorflow as tf
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I hate you")[0]

print(f"label: {result['label']}, with score: {round(result['score'], 4)}")


# Breaking down the pipeline function
from transformers import AutoTokenizer
from transformers import TFAutoModel

checkpoint = 'distilbert-base-uncased-finetuned-sst-2-english'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = ['Life is too beautiful to waste on negativity',
              'Arise, Awake, stop not till the goal is reached']

inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors='tf')
print(inputs)

model = TFAutoModel.from_pretrained(checkpoint)
outputs = model(**inputs)
outputs.keys()
outputs['last_hidden_state'].shape

    # Since we are doing sequence classification here
from transformers import TFAutoModelForSequenceClassification
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(inputs)

outputs.keys()
print(outputs['logits'])

predictions = tf.math.softmax(outputs.logits, axis=-1)
print(predictions)
model.config.id2label