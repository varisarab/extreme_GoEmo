
# %pip install datasets transformers onnx onnxruntime

#Use pre-trained https://huggingface.co/microsoft/xtremedistil-l6-h256-uncased model, fine-tune on the emotion classification.

model_name = 'microsoft/xtremedistil-l6-h384-uncased'
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

from datasets import load_dataset
ds = load_dataset("go_emotions", "raw")

emotions = [
 'admiration',
 'amusement',
 'anger',
 'annoyance',
 'approval',
 'caring',
 'confusion',
 'curiosity',
 'desire',
 'disappointment',
 'disapproval',
 'disgust',
 'embarrassment',
 'excitement',
 'fear',
 'gratitude',
 'grief',
 'joy',
 'love',
 'nervousness',
 'optimism',
 'pride',
 'realization',
 'relief',
 'remorse',
 'sadness',
 'surprise',
 'neutral']

ds = ds.map(lambda x : {"labels": [x[c] for c in emotions]})

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

cols = ds["train"].column_names
cols.remove("labels")
ds_enc = ds.map(tokenize_function, batched=True, remove_columns=cols)
ds_enc

import torch
ds_enc.set_format("torch")
ds_enc = (ds_enc
          .map(lambda x : {"float_labels": x["labels"].to(torch.float)}, remove_columns=["labels"])
          .rename_column("float_labels", "labels"))

ds_enc['train'].features


#define model and parameters
#import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(emotions), problem_type="multi_label_classification")
model = model.to(device)

from transformers import TrainingArguments
training_args = TrainingArguments("test_trainer",
                                  per_device_train_batch_size=128, 
                                  num_train_epochs=4,learning_rate=3e-05,
                                  evaluation_strategy="no")
from transformers import Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_enc['train']
)

trainer.train()

#ONNX Runtime Web: save to ONNX format

import transformers
import transformers.convert_graph_to_onnx as onnx_convert
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bergum/xtremedistil-l6-h384-go-emotion")
model = AutoModelForSequenceClassification.from_pretrained("bergum/xtremedistil-l6-h384-go-emotion")

pipeline = transformers.pipeline("text-classification",model=model,tokenizer=tokenizer)

pipeline("I am content")

?onnx_convert.convert_pytorch(pipeline, opset=11, output=Path("extreme-go-emotion.onnx"), use_external_format=False)

from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("extreme-go-emotion.onnx", "extreme-go-emotion-int8.onnx", 
                 weight_type=QuantType.QUInt8)

from google.colab import files

files.download("extreme-go-emotion-int8.onnx")

model = model.to("cpu")

!apt-get install git-lfs

token="huggingfacetoken"

model.push_to_hub("extreme-go-emotion", use_auth_token=token)

tokenizer.push_to_hub("extreme-go-emotion", use_auth_token=token)

