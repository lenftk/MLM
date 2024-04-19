from tokenizers import CharBPETokenizer

tokenizer = CharBPETokenizer(suffix='', lowercase=True)

special_tokens = ['<pad>','<unk>','<s>','</s>','<mask>']

vocab_size = 36000
min_frequency = 2

data_path = 'C:/Users/USER/Desktop/Project_LLM/data/wikitext-2-raw'
tokenizer.train(files=data_path + '/wiki.train.raw',
    vocab_size=vocab_size,
    min_frequency=min_frequency,
    special_tokens=special_tokens,
    suffix='')

tokenizer.save('tokenizer.json')