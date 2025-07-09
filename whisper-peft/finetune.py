from datasets import load_dataset, DatasetDict
from sagemaker.inputs import TrainingInput
from peft import LoraConfig, get_peft_model


def prepare_dataset(batch):
  audio = batch["audio"]
  batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
  batch["labels"] = tokenizer(batch["sentence"]).input_ids
  return batch


def __init__():
    language = "Marathi"
    language_abbr = "mr"
    task = "transcribe"
    dataset_name = "mozilla-foundation/common_voice_11_0"
    
    common_voice = DatasetDict()
    common_voice["train"] = load_dataset(dataset_name, language_abbr, split="train+validation", use_auth_token=True)
    common_voice["test"] = load_dataset(dataset_name, language_abbr, split="test", use_auth_token=True)
    
    
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
    tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=language, task=task)
    
    
    
    
    common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=2)
    common_voice.save_to_disk("marathi-common-voice-processed")
    ##!aws s3 cp --recursive "marathi-common-voice-processed" s3://<Your-S3-Bucket>
    
    training_input_path=s3uri
    training = TrainingInput(
    s3_data_type='S3Prefix', # Available Options: S3Prefix | ManifestFile | AugmentedManifestFile
    s3_data=training_input_path,
    distribution='FullyReplicated', # Available Options: FullyReplicated | ShardedByS3Key
    input_mode='FastFile'
    )
    
    
    config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
    model = get_peft_model(model, config)
    
    training_args = Seq2SeqTrainingArguments(
    output_dir=args.model_dir,
    per_device_train_batch_size=int(args.train_batch_size),
    gradient_accumulation_steps=1,
    learning_rate=float(args.learning_rate),
    warmup_steps=args.warmup_steps,
    num_train_epochs=args.num_train_epochs,
    evaluation_strategy="epoch",
    fp16=True,
    per_device_eval_batch_size=args.eval_batch_size,
    generation_max_length=128,
    logging_steps=25,
    remove_unused_columns=False,
    label_names=["labels"],
    )
    trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset["train"],
    eval_dataset=train_dataset.get("test", train_dataset["test"]),
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
    )

    OUTPUT_PATH= f's3://{BUCKET}/{PREFIX}/{TRAINING_JOB_NAME}/output/'

    huggingface_estimator = HuggingFace(entry_point='train.sh',
    source_dir='./src',
    output_path= OUTPUT_PATH,
    instance_type=instance_type,
    instance_count=1,
    # transformers_version='4.17.0',
    # pytorch_version='1.10.2',
    py_version='py310',
    image_uri=<ECR-PATH>,
    role=ROLE,
    metric_definitions = metric_definitions,
    volume_size=200,
    distribution=distribution,
    keep_alive_period_in_seconds=1800,
    environment=environment,
    )

    huggingface_estimator.fit(job_name=TRAINING_JOB_NAME, wait=False)
    metric = evaluate.load("wer")

    eval_dataloader = DataLoader(common_voice["test"], batch_size=8, collate_fn=data_collator)
    
    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
      with torch.cuda.amp.autocast():
        with torch.no_grad():
          generated_tokens = (model.generate(input_features=batch["input_features"].to("cuda"),decoder_input_ids=batch["labels"][:, :4].to("cuda"),max_new_tokens=255,).cpu().numpy())
      labels = batch["labels"].cpu().numpy()
      labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
      decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
      decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
      metric.add_batch(predictions=decoded_preds,references=decoded_labels,)
    del generated_tokens, labels, batch
      gc.collect()
      wer = 100 * metric.compute()
      print(f"{wer=}")


    
    
    
