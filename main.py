import tokenizers
from transformers import BertConfig, BertForMaskedLM
from transformers import BertTokenizer
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer

corpus_path = './corpus/sentences.txt'
vocab_save_path = './tokenizer/'
tokenizer_save_path = './tokenizer/tokenizer.json'
tokenizer_load_path = './tokenizer/'
pretrained_models_path = './pretrained_models/'


def train_tokenizer():
    bwpt = tokenizers.BertWordPieceTokenizer()
    bwpt.train(
        files=[corpus_path],
        vocab_size=50000,
        min_frequency=3,
        limit_alphabet=1000
    )
    bwpt.save_model(vocab_save_path)
    bwpt.save(tokenizer_save_path)


def get_tokenizer():
    tokenizer = BertTokenizer.from_pretrained(tokenizer_load_path)
    return tokenizer


def check_tokenizer(tokenizer):
    sentence = '12 314 162 315 62 276'
    encoded_input = tokenizer.tokenize(sentence)
    print(encoded_input)


def get_dataset_and_data_collator(tokenizer):
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=corpus_path,
        block_size=128  # maximum sequence length
    )
    print('Number of lines in dataset: ', len(dataset))
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    return dataset, data_collator


def get_bert_model():
    config = BertConfig(
        vocab_size=50000,
        num_hidden_layers=6,
    )
    model = BertForMaskedLM(config)
    print('Number of parameters: ', model.num_parameters())
    return model


def get_trainer(dataset, data_collator, model):
    training_args = TrainingArguments(
        output_dir=pretrained_models_path,
        overwrite_output_dir=True,
        per_device_train_batch_size=32,
        num_train_epochs=50,
        save_steps=10000,
        save_total_limit=3,
        fp16=True,
    )
    trainer = Trainer(
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        model=model,
    )
    return trainer


def train():
    tokenizer = get_tokenizer()
    dataset, data_collator = get_dataset_and_data_collator(tokenizer)
    model = get_bert_model()
    trainer = get_trainer(dataset, data_collator, model)
    trainer.train()
    trainer.save_model(pretrained_models_path)


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    train()
