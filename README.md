# NLP_assignment1

# Task 1. Tokenize some tweets manually (20 points)
As a first task you need to tokenize first 15 tweets from file2 by hand. This will allow you to understand the problem from a linguistic point of view. The guidelines for tweet tokenization are as follows:

# Task 2. Implement Byte-Pair Encoding(BPE) Tokenizer (80 points)
## Task 2.1. Implementation (60 points)
Implement the tokenizer as the BPETokenizer class:

Implement train method that learns merges and builds the vocabulary of the specified vocab_size (25 points).
Implement tokenize method that should tokenize the text according to the learnt merges (25 points).
Your code should have docstrings and comments (10 points).

## Task 2.2. Analysis on Tweets Dataset (10 points)
Train the BPE tokenizer on the tweets dataset. Try to tokenize the tweets with the tokenizer of different vocab_size. For example, train the BPE tokenizer with vocab_size of [base_vocab_size, 250, 500, 750, 1000]. Plot the dependency of the average length of the tokenized tweet by vocab_size to analyze how vocab_size affects the length of the tokenized tweet on average. Tell what vocab_size is preferrable and why.
![загруженное](https://github.com/mpvnlv/NLP_assignment1/assets/88908152/3874e83e-1349-465d-9628-4f921a898778)

Byte Pair Encoding (BPE) is a tokenization method that is used to partition text into subwords. When vocab_size is increased in BPE, we increase the number of subwords that the model can use to represent the text.

When vocab_size is increased, BPE will combine word parts into larger subwords to fit the new vocabulary size. This can cause the same word parts to be merged together to create new tokens. As a result, we will have fewer tokens to represent the same text.

For example, with vocab_size=1000, the word "book printing" can be split into "book" and "printing", while with vocab_size=2000, the same word can be split into "kn" and "igoprinting". Thus, as vocab_size increases, the number of tokens per sentence may decrease due to the larger subwords used to represent the text.
## Task 2.3. Analysis on Dataset of Different Language (10 points)
Find a small dataset of texts in a language other than English. The dataset size should be not greater than several megabytes.

Train the BPE tokenizer on the dataset that you found. Try to tokenize the sentences from this dataset with the tokenizer of different vocab_size. Plot the dependency of the average length of the tokenized sentence by vocab_size to analyze how vocab_size affects the length of the tokenized sentence on average.

Tell how how the average length of the tokenized sentence differs from the average length of the tokenized tweet. Explain why.
![image](https://github.com/mpvnlv/NLP_assignment1/assets/88908152/9f7400c6-ac2c-41a5-83c7-e193949a7fba)


