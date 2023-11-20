from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
import shap
import numpy as np
import transformers
import torch
from shap.maskers import Text
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datasets import Dataset
import json, argparse

parser = argparse.ArgumentParser(description="Your Script Description")

parser.add_argument("--device", type=str, help="Device name")
parser.add_argument("--base-dir", type=str, help="Base directory")
parser.add_argument("--model-dir", type=str, help="Model directory")
parser.add_argument("--data-dir", type=str, help="Data directory")
parser.add_argument("--data-type", type=str, choices=["pandas", "Dataset"], help="Type of data")
parser.add_argument("--mask-token", type=str, help="Mask token", default="<mask>")
parser.add_argument("--task-name", type=str, help="Name of this task")

args = parser.parse_args()
device = torch.device(args.device)


class model_wrapper(torch.nn.Module):
    def __init__(self, model, tokenizer, device):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model.to(device)
        self.device = device
    def forward(self, texts):
#         n_samples = len(texts)
#         labels = ['Benign', "HT Risk"]
        texts = list(texts)
        # Tokenize and encode the input text
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        # Make predictions with your model
        with torch.no_grad():
            outputs = self.model(**inputs)
#         scores = torch.softmax(outputs.logits, dim=1)
#         output = []
#         for i in range(n_samples):
#             max_inx = torch.argmax(scores[i])
#             output.append({"label": labels[max_inx], "score": scores[i,max_inx].detach().numpy().item()})
        # For text classification, you might want to return the class probabilities
#         return output
        return torch.softmax(outputs.logits, dim=1)


additional_tokens = [
        "[PHONE]",
        "[NAME]",
        "[LOCATION]",
        "[ONLYFANS]",
        "[SNAPCHAT]",
        "[USERNAME]",
        "[INSTAGRAM]",
        "[TWITTER]",
        "[EMAIL]"
    ]
model_path = f"{args.base_dir}/{args.model_dir}"
model_name = model_path.split('/')[-1]
data_path = f"{args.base_dir}/{args.data_dir}"
figure_path = Path(f"{args.base_dir}/shap_results/{model_name}/figures")
figure_path.mkdir(parents=True, exist_ok=True)
html_path = Path(f"{args.base_dir}/shap_results/{model_name}/html")
html_path.mkdir(parents=True, exist_ok=True)
dict_path = Path(f"{args.base_dir}/shap_results/{model_name}/dict")
dict_path.mkdir(parents=True, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, additional_special_tokens=additional_tokens, max_length=512)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(
    model_path, num_labels=2, ignore_mismatched_sizes=True
)
model.resize_token_embeddings(len(tokenizer))
model.config.problem_type = "single_label_classification"

model = model_wrapper(model, tokenizer, device)

if(args.data_type == "pandas"):
    df = pd.read_json(data_path)
elif(args.data_type == "Dataset"):
    df = Dataset.from_json(data_path).to_pandas()
selected_post_text = list(df['text'])
selected_post_label = list(df['label'])

masker = Text(tokenizer, mask_token=args.mask_token)
explainer = shap.Explainer(model, masker)
shap_values = explainer(selected_post_text)
res = {'records': []}
for i in range(len(shap_values.data)):
    data = {}
    data['text'] = df['text'][i]
    data['label'] = int(df['label'][i])
    data['post_id'] = int(df['post_id'][i])
    data['tokens'] = shap_values.data[i].tolist()
    data['attributions'] = shap_values.values[i].tolist()
    res['records'].append(data)
with open(f"{dict_path}/{args.task_name}.json", "w") as outfile: 
    json.dump(res, outfile)
html = shap.plots.text(shap_values[:, :, 1], display=False)
f = open(f"{html_path}/{args.task_name}.html", "w")
f.write(html)
f.close()

shap.plots.bar(shap_values[0, :, 1], max_display=20, show=False)
plt.savefig(f"{figure_path}/{args.task_name}.pdf")
