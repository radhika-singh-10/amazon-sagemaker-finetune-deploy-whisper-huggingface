import json
import time
import boto3
import numpy as np
import sagemaker
import sagemaker.huggingface
import os
from peft import PeftModel, PeftConfig
from transformers import WhisperForConditionalGeneration, Seq2SeqTrainer
from huggingface_hub import snapshot_download
from sagemaker.huggingface import HuggingFace
from sagemaker.inputs import TrainingInput
from datasets import load_from_disk
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import WhisperProcessor, WhisperTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import evaluate
import gc







ROLE = sagemaker.get_execution_role()
sess = sagemaker.Session()
BUCKET = sess.default_bucket()
PREFIX = "whisper/data/marathi-common-voice-processed"
s3uri = os.path.join("s3://", BUCKET, PREFIX)
print(f"sagemaker role arn: {ROLE}")
print(f"sagemaker bucket: {BUCKET}")
print(f"sagemaker session region: {sess.boto_region_name}")
print(f"data uri: {s3uri}")


distribution = None
instance_type = 'ml.g5.2xlarge'
training_batch_size = 16
eval_batch_size = 8



snapshot_download(repo_id="openai/whisper-large-v2", local_dir="/tmp/whisper-large-v2/")


os.system('aws s3 cp --recursive "/tmp/whisper-large-v2" s3://YOUR_BUCKET/whisper-large-v2/pretrain/')



id = int(time.time())
TRAINING_JOB_NAME = f"whisper-mr-{id}"
print('Training job name: ', TRAINING_JOB_NAME)

model_name_s3 = "whisper-large-v2"
environment = {
    'MODEL_S3_BUCKET': BUCKET,
    'MODEL_NAME_S3': model_name_s3,
    'DATA_S3': s3uri,
}

metric_definitions = [
    {'Name': 'eval_loss', 'Regex': "'eval_loss': ([0-9]+(.|e\\-)[0-9]+),?"},
    {'Name': 'eval_wer', 'Regex': "'eval_wer': ([0-9]+(.|e\\-)[0-9]+),?"},
    {'Name': 'eval_runtime', 'Regex': "'eval_runtime': ([0-9]+(.|e\\-)[0-9]+),?"},
    {'Name': 'eval_samples_per_second', 'Regex': "'eval_samples_per_second': ([0-9]+(.|e\\-)[0-9]+),?"},
    {'Name': 'epoch', 'Regex': "'epoch': ([0-9]+(.|e\\-)[0-9]+),?"}
]



training_input_path = s3uri
training = TrainingInput(
    s3_data_type='S3Prefix',
    s3_data=training_input_path,
    distribution='FullyReplicated',
    input_mode='FastFile'
)


OUTPUT_PATH = f's3://{BUCKET}/{PREFIX}/{TRAINING_JOB_NAME}/output/'

huggingface_estimator = HuggingFace(
    entry_point='train.sh',
    source_dir='./src',
    output_path=OUTPUT_PATH,
    instance_type=instance_type,
    instance_count=1,
    py_version='py310',
    image_uri='348052051973.dkr.ecr.us-east-1.amazonaws.com/whisper:training',
    role=ROLE,
    metric_definitions=metric_definitions,
    volume_size=200,
    distribution=distribution,
    keep_alive_period_in_seconds=1800,
    environment=environment,
)


huggingface_estimator.fit(job_name=TRAINING_JOB_NAME)
os.system('./s5cmd sync s3://YOUR_BUCKET/whisper-large-v2/output/SOME_DATE/whisper_out/adapter_model/ adapter_model/')
os.system('./s5cmd sync s3://YOUR_BUCKET/whisper-large-v2/pretrain/ /tmp/whisper-large-v2/')



peft_model_id = "adapter_model"
peft_config = PeftConfig.from_pretrained(peft_model_id)
model = WhisperForConditionalGeneration.from_pretrained(
    peft_config.base_model_name_or_path, load_in_8bit=True, device_map="auto"
)
model = PeftModel.from_pretrained(model, peft_model_id)
model.config.use_cache = True

common_voice = load_from_disk("marathi-common-voice-processed")



@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch



model_name_or_path = "/tmp/whisper-large-v2"
language = "Marathi"
task = "transcribe"
tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=language, task=task)
processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)



metric = evaluate.load("wer")
eval_dataloader = DataLoader(common_voice["test"], batch_size=8, collate_fn=data_collator)

model.eval()
for step, batch in enumerate(tqdm(eval_dataloader)):
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            generated_tokens = model.generate(
                input_features=batch["input_features"].to("cuda"),
                decoder_input_ids=batch["labels"][:, :4].to("cuda"),
                max_new_tokens=255,
            ).cpu().numpy()

            labels = batch["labels"].cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    del generated_tokens, labels, batch
    gc.collect()

wer = 100 * metric.compute()
print(f"{wer=}")
