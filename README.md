# NLP_assignment1

# Task 1. Tokenize some tweets manually (20 points)
As a first task you need to tokenize first 15 tweets from file2 by hand. This will allow you to understand the problem from a linguistic point of view. 
1. Input tweet
   Camping in Maine for the weekend. Hey Dad, Mama Loves YOU: http://www.mamapalooza.com
1. Tokenized tweet
   Camping, in, Maine, for, the, weekend, ., Hey, Dad, , , Mama, Loves, YOU, :,  http://www.mamapalooza.com
2. Input tweet
   Its american tradition bitch
2. Tokenized tweet
   Its, american, tradition, bitch
3. Input tweet
   @ThroughTheVoid They love it! The only pleasure they get in life. I actually do that. I'm sure I hear a tiny squeak... Then louder ones
3. Tokenized tweet
   @ThroughTheVoid, They, love, it, !, The, only, pleasure, they, get, in, life, ., I, actually, do, that, ., I'm, sure, I, hear, a, tiny, squeak, ... , Then, louder, ones
4. Input tweet
   " RT @latti: @AbsoHilare stop tweeting in church! Lol <--- ""I tweet because I'm happy, I tweet because I'm free"" LOL!"
4. Tokenized tweet
   ", RT, @latti, :, @AbsoHilare, stop, tweeting, in, church, !, Lol, <---, ", ", I, tweet, because, I'm, happy, ,, I, tweet, because, I'm, free, ",", LOL, ! , "
5. Input tweet
   Samsung Mini S2 portable HDD graced with colors that perfectly match your tacky beach gear: Sammy's done it aga.. http://tinyurl.com/lb5p6m
5. Tokenized tweet
   Samsung, Mini, S2, portable, HDD, graced, with, colors, that, perfectly, match, your, tacky, beach, gear, :, Sammy's,  done, it, aga, ..,  http://tinyurl.com/lb5p6m
6. Input tweet
 @dialloc congrats on finding your way over. it may be slow going at first. hang in there. it's kinda cool when u get up to speed.
6. Tokenized tweet
 @dialloc, congrats, on, finding, your, way, over, ., it, may, be, slow, going, at, first, ., hang, in, there, ., it's, kinda, cool, when, u, get, up, to, speed, .
7. Input tweet
 iPhone activation delays continue, Apple offers $30 http://twt.gs/l3Ki
7. Tokenized tweet
 iPhone, activation, delays, continue, ,, Apple, offers, $30, http://twt.gs/l3Ki
8. Input tweet
 RT @GoogleAtWork Gmail maximum attachment size now 25MB http://bit.ly/62mjw Nice!!!
8. Tokenized tweet
 RT, @GoogleAtWork, Gmail, maximum, attachment, size, now, 25MB, http://bit.ly/62mjw Nice, !!!
9. Input tweet
 RT @acfou The Ads Won Awards for Crispin; But Did Nothing for Client BurgerKing's Sales/Marketshare - Big Surprise - http://ping.fm/vw8TI
9. Tokenized tweet
 RT, @acfou, The, Ads, Won, Awards, for, Crispin, ;, But, Did, Nothing, for, Client, BurgerKing's, Sales/Marketshare, -, Big, Surprise, -, http://ping.fm/vw8TI
10. Input tweet
 Hey doll! Great I missed True Blood yday boo lol Rt @FrankBanuat78 @jhillstephens Hello Sunshine how are u today? :-)
10. Tokenized tweet
 Hey, doll, !, Great, I, missed, True, Blood, yday, boo, lol, Rt, @FrankBanuat78, @jhillstephens, Hello, Sunshine, how, are, u, today, ?, :-)
11. Input tweet
 Australian artist Pogo made these free songs primarily from sampled audio from Alice In Wonderland. http://www.last.fm/music/Pogo/Wonderland
11. Tokenized tweet
 Australian, artist, Pogo, made, these, free, songs, primarily, from, sampled, audio, from, Alice, In, Wonderland, ., http://www.last.fm/music/Pogo/Wonderland
12. Input tweet
 @mppritchard they wanted to sell all the preorders & then sell all of the ones they had in stock to those that just walked in. Can't do both
12. Tokenized tweet
 @mppritchard, they, wanted, to, sell, all, the, preorders, &, then, sell, all, of, the, ones, they, had, in, stock, to, those, that, just, walked, in, ., Can't, do, both
13. Input tweet
 Incoming: Frightened Rabbit, Sept. 22 (Tucson): If Fat Cat Records is going to send three great bands from Scot.. http://tinyurl.com/nz6xcv
13. Tokenized tweet
 Incoming, :, Frightened, Rabbit, ,, Sept, ., 22, (Tucson), :,  If, Fat, Cat, Records, is, going, to, send, three, great, bands, from, Scot.. ,http://tinyurl.com/nz6xcv
14. Input tweet
 Hey @ginoandfran please greet philip! (GinoandFran live > http://ustre.am/2YyQ)
14. Tokenized tweet
 Hey, @ginoandfran, please, greet, philip, !, (, GinoandFran, live, >, http://ustre.am/2YyQ, )
15. Input tweet
 Ik weet niet wie er achter de T-Mobile iPhone Twitter zit maar ik vind het niet echt 'corporate' taalgebruik... Best vreemd eigenlijk
15. Tokenized tweet
 Ik, weet, niet, wie, er, achter, de, T-Mobile, iPhone, Twitter, zit, maar, ik, vind, het, niet, echt, 'corporate', taalgebruik, ..., Best, vreemd, eigenlijk

# Task 2. Implement Byte-Pair Encoding(BPE) Tokenizer (80 points)
## Task 2.1. Implementation (60 points)
Implement the tokenizer as the BPETokenizer class:
  Implement train method that learns merges and builds the vocabulary of the specified vocab_size (25 points).
  Implement tokenize method that should tokenize the text according to the learnt merges (25 points).
Your code should have docstrings and comments (10 points).

Implementation of this task in BPE_tokenizer.ipynb in this repo

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


