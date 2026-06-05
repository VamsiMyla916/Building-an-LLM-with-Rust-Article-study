# Learnings so far from the [Building LLM from scratch using Rust](https://www.tag1.com/how-to/part1-tokenization-building-an-llm-from-scratch-in-rust/)

Tokenization, building an LLM from scratch using Rust.

# Part 1 TOKENIZER:

3 things:

1. Converting Text to numbers using Byte Pair Encoding (BPE)
2. How this step shapes everything the model can and cannot do.
3. How different vocabulary sizes effect compression, speed and training behaviour.

Questions that popped up during the study:

1. Tokenizer? -> Tool used to convert text to numbers
2. BPE? Algorithms used in the tokenization process
3. Feste? -> The LLM name that we are building
4. Transformer? -> Deep learning architecture
5. Shakesphere? -> Real one? or any name for a tech/software? -> The Real one! and this model is trained on Shakesphere complete works.

## Learning about Transformer

Question: What exactly is a transformer and difference between transformer and a language model?

Transformer is a deep learning architecture used in the language model. It basically has encoding-decoding structure and works on the "Attention Mechanism"

## What is attention mechanism?

This method finds the relation between multiple words and how are they related with each other in a sentence.

Difference between transformer and a conventional NN: This Transformer is found as a replacement for RNN/CNN. The conventional neural networks process data sequentially which takes a lot of time and storage space where as Transformer model can process large amount of data points parallelly like finding relevance, encoding, decoding etc.

## More about the transformer model:

Transformer model holds a large amount of context. Transformer model does paralelly processes parameters like grammar, emotion etc. It has multi-head self attention for finding and understanding relationship between words. It also does positional encoding to maintain the order of words in a sentence.

Example: Ramu goes to temple in his village. In this sentence the word "his" refers to the person name "Ramu". This is identified by the model using the self attention mechanism.

## Applications:

### Training a customer support chatbot, medical report summaries etc

## Benefits:

1. Long range context

1. Concurrency/Parallelism

1. Scalability

Resources: [About Transformers](https://youtu.be/lopXj1p6Ewk?si=XnRMJLDPw4PwhJrm)

## What are we building?

#### 'Feste' -> A GPT 2 Style Architecture model built in complete Rust as in general using Tensorflow/PyTorch for building transofrmer model abstracts away (eliminates or gives a theoretical level idea) math so that we can focus on the architecture. But in Rust it includes low-level memory management, performance, borrow checker on ownership and memory (control over all these things) safety.

# Why Tokenization matter?

## Considering the following example:

["he", "llo"] -> say "he" represents tokenID 530 and "llo" tokenID 840 respectively

[" he","llo "] -> say " he" represents tokenID 460 and "llo " tokenID 670 respectively

["ol", "leh"] -> say "ol" represents tokenID 91 and "leh" represents tokenID 133 respectively

Though the memory contains the word "hello", it can't reverse the string because of different tokenIDs in each scenario resulting in a mismatch. This is one of the reasons a simple LLM stuggles to perform operations on a string like reversing it. At word level tokenization is not efficient.

# Algorithm used for Tokenization by Feste is BPE

The goal is to convert Text to TokenIDs -> process neural networks -> convert those tokenIDs back to text and generate the output.

Instead of using tokenization at the word level we would go to character level where each character is a token.

Byte level -> 8bit (binary digits) represent exactly 256 values.

English characters use 1 byte each while others use multiple bytes. For example emojis use 4bytes.

Each byte gets it's own tokenID in the file

[Why 256?](https://youtu.be/4pcXW7l-IKU?si=4UmJrN4x1KVtblxB)

### BPE Assigning

| Character | Byte | Token ID |
| :-------- | :--- | :------- |
| A         | 65   | 530      |
| B         | 66   | 870      |
| C         | 67   | 754      |
| D         | 68   | 214      |
| E         | 69   | 125      |
| F         | 70   | 859      |
| G         | 71   | 381      |
| H         | 72   | 350      |
| I         | 73   | 328      |
| J         | 74   | 242      |
| K         | 75   | 854      |
| L         | 76   | 204      |
| M         | 77   | 792      |
| N         | 78   | 858      |
| O         | 79   | 658      |
| P         | 80   | 189      |
| Q         | 81   | 704      |
| R         | 82   | 532      |
| S         | 83   | 132      |
| T         | 84   | 130      |
| U         | 85   | 195      |
| V         | 86   | 323      |
| W         | 87   | 338      |
| X         | 88   | 617      |
| Y         | 89   | 716      |
| Z         | 90   | 127      |

### Iteratively merges the most frequent adjacent tokens to new tokens

Example: "th" is frequently repeated and "t" and "h" has individual tokens assigned initially. Since they are a frequently repeated pair they are merged together and gets asigned with a single new token say 344.

Common words end up with own tokens whereas rare words break down into familiar byte sequences that tokenizer seen in other context.

Smaller vocabulary -> fragmentation building words from small pieces. Larger vocabulary -> more to find meaning if trained well enough and identify conceptual relation with other words that are generated next

# Implementation Details:

Starting with 256 entries mapping hex-encoded bytes to IDs (0-255)

During training -> add entries for each merge

Merge rules stored in a vector for order significance

# Example

Before encoding "the" -> should apply merge1(M1) first for token256 then apply merge50 (M50)

M1 -> "th" <74><68> token 256 ...... M50 ->"the" => token 256+<65> = token 300

Training process is completely deterministic like count pairs, find frequently occurring pairs, merge them, repeat. This ensures in no randomness.

# Training Performance

Bottleneck: Counting adjacent pairs in the corpus

Instead of single threaded implementation, using Rayon(Rust library) for parallel counting where chunks (chunk1,chunk2...etc) are divided and counted in parallel.

Each thread checks its chunk boundary and count pair between last token and next chunk's first token such that no pairs are missed.

Sequential: Apply merge rules and updating corpus

Parallelism: Chunk boundary checks so that no pair is missed.

# Parallelization using Rayon (Rust lib)

### Parallel Processing Boundary Check Diagram

Below is a diagram illustrating how separate threads process distinct text chunks in parallel. The crucial step highlighted is the "Boundary Check," where Thread 1 looks at its last formed token and Thread 2's first formed token to identify pairs that might span across the chunk break (preventing missed merges like pre-existing "th" or "the" tokens).

thread 1 ->[["t","h"] ["e","a","d"]] , thread 2 ->[["u","s"] ["e","d","t"]] .... thread n [[][]]

Basically this is a parallel processing method for multi-thread operations where each thread checks the chunk boundary to see if it is missing any pairs to merge like "th" and "the" (from thread 1) and "us", "use", "used" (from thread 2)

This multi-thread operation helps in speeing up the process. Each thread checks the chunk boundaries so that no pair gets missed out. We only focus on the bottleneck which is counting pairs and not on speeding up the max performance like applying merges or updating the corpus as optimizing them complicates the code.

# Encoding

Consider the following example:

Vocabulary: "To be"

Their hex values are as follows
| Character | Hex Value |
| :--- | :--- |
| T | 54 |
| o | 6f |
| Space | 20 |
| b | 62 |
| e | 65 |

Now the list is : [<54>,<6f>,<20>,<62>,<65>]

Say the model learned 3 merges in the order:

M1: <6f><20> -> "o " -> Token 256

M2: <54><6f> -> "To" -> Token 257

M3: Token 256 + <62> -> "o b" -> 258

Encoding follows a sequential order while applying these merge rules:

From M1, the list will be updated from [<54>,<6f>,<20>,<62>,<65>] to [<54>,256,<62>,<65>]

From M2, the list will remain same because in this case, it looks for applying the M2 to the updated list, but it can't find the hex value <6f> as it was updated to token 256 during the M1.

From M3, it looks for Token 256 and hex <62> and both exist in the newly updated list. So this rule will be applied to the list and now it becomes [<54>,258,<65>]

To summarize the encoding process, basically the model learns some merge rules and apply them sequentially to a list. If merge rules aligns with that list then the list is updated, if a merge rule could not find the token/hex value to apply the merge then the list won't get updated and will remain the same and it will skip to the next merge rule.

# Decoding

| Token | Map to | Hex Byte     | Expected Output |
| :---- | :----: | :----------- | :-------------- |
| <54>  |   ->   | <54>         | T               |
| 258   |   ->   | <6f><20><62> | o b             |
| <65>  |   ->   | <65>         | e               |

After mapping the token IDs back to their corresponding hex byte values, join all those hex bytes together <54><6f><20><62><65> and parse them back to the text ("To be") as per the UTF-8 standards.

Important Note: For this educational model, the chunks are processed independently and the "Loss in compression efficiency is neglegible". BPE follows strict guidelines to prevent breaking boundary between chunks (Ex: "Formatting" might split into "Format" "ting")

# Training Results:

- In this section, the article mentioned about a model that can train 5 different tokenizers using vocabulary sizes of 256,512,1024,1536,20534 respectively.

- Question: How to run this model?
- Production models use 20534 tokens (GPT2+GPT3 -> 50257 tokens)

- Compression Ration: How many types of input each token represents on average.

- CR = Total Bytes of text/ Total Tokens used

- If CR is 1.96x then one token represent 1.96 bytes.
- Shortersequences and smaller vocabulary means faster processing and less data processing during training and inference.

- Uses less memory

- Embedding matrix -> digital dictionary to hold token values where each row represents a token and token is represented by decimal numbers called embedding parameters. These decimal values identify the semantic relationship between two tokens.

### Token Embedding Matrix (Parameter Lookup Table)

Once tokens are assigned their integer IDs, they are mapped to high-dimensional vectors. These floating-point numbers are the actual parameters (weights) the model learns during training to understand relationships between tokens.

| Token ID  | Token String      | $d_0$   | $d_1$   | $d_2$   | $d_3$   | $\dots$ | $d_{511}$ |
| :-------- | :---------------- | :------ | :------ | :------ | :------ | :------ | :-------- |
| **256**   | `th`              | 0.1245  | -0.4512 | 0.8821  | -0.0192 | $\dots$ | -0.0034   |
| **257**   | `To`              | -0.9921 | 0.0154  | 0.3409  | 0.1125  | $\dots$ | 0.7710    |
| **258**   | `o b`             | 0.3310  | -0.2100 | -0.5541 | 0.9982  | $\dots$ | 0.1093    |
| **300**   | `the`             | 0.5501  | 0.2219  | -0.1123 | -0.7761 | $\dots$ | -0.4428   |
| **...**   | ...               | ...     | ...     | ...     | ...     | $\dots$ | ...       |
| **50256** | `<\|endoftext\|>` | -0.1102 | 0.8842  | 0.0012  | -0.3341 | $\dots$ | 0.0001    |

**Note on the dimensions:** \* **Rows:** Equal to the Vocabulary Size (e.g., 50,257 rows for GPT-2).

- **Columns ($d_n$):** The embedding dimension or hidden size of the model (e.g., 512, 768, or 4096).
- For vocabulary 256 say each token might have 768 parameters then the total embedded parameters will be 256 \* 768 = 196608
- Better compression requires checking of more merge rules.

## Comparision Table to identify the Compression Ratio and Training requirements across different vocabulary sizes.

| Vocabulary Size | Encoded Length (tokens) | Compression Ratio | Training Time (s) | Encoding Time (s) |
| :-------------- | :---------------------- | :---------------- | :---------------- | :---------------- |
| 256             | 5,422,721               | 1.00x             | 0.00              | 0.14              |
| 512             | 2,772,080               | 1.96x             | 25.54             | 2.04              |
| 1024            | 2,189,778               | 2.48x             | 77.45             | 4.88              |
| 1536            | 1,952,470               | 2.78x             | 142.51            | 7.46              |
| 20,534          | 1,481,106               | 3.66x             | 286.34            | 242.50            |

- Higher the Compression Ration, better the output. But Encoding cost will be higher and also requires checking of more merge rules.

| Vocabulary Size | Base Vocabulary | Additional Merges | Total Merges / Breakdown                     |
| :-------------- | :-------------- | :---------------- | :------------------------------------------- |
| **256**         | 256             | 0                 | $256 \text{ (Base)} + 0 \text{ merges}$      |
| **512**         | 256             | 256               | $256 \text{ (Base)} + 256 \text{ merges}$    |
| **1024**        | 256             | 768               | $256 \text{ (Base)} + 768 \text{ merges}$    |
| **1536**        | 256             | 1280              | $256 \text{ (Base)} + 1280 \text{ merges}$   |
| **20,534**      | 256             | 20,278            | $256 \text{ (Base)} + 20,278 \text{ merges}$ |

- Phrase-level compression happens for few phrases at 20534 size vocabulary.

# Trade-offs

- Compression Ration increases with increase in vocabulary size but with diminishing returns.
- My Assumption based on learnings so far: Reason could be since the packing rate is happening for frequently repeating phrases/letters/words and those that doesn't repease often remain the same.
- Training time increases almost linearly with vocabulary size until sampling optimization kicks in.

## Data Scarcity problem

- If we choose a vocabulary size say 1536, but the dataset os too small, then the model forces to remember small words which does not repeat often or appears only once.
- This also leads to embedding collapse and treat arare words as random noise.

## Solution: Choosing the right vocabulary

- Choosing the right vocabulary size would fix this problem.
- Smaller vocabularies result in larger sequences which makes training slower.
- But in the Trade-offs sections, as per the visuals mentioned in the article, the training times increases with vocabulary size (almost linearly) if sampling optimization is not implemented.

# What we chose not to do?

- Choosing clarity over performance.
- Understanding how tokenization works>>>>> than trying for extra speed.
- Some optimizations by production tokenizers aren't really necessary.

# Encoding Performance

- Merging follows sequence.
- Merging increases with Vocabulary size.
- Production tokenizers use trie-based lookup or cache encoding to speed up this process.
- It is mentioned in the article that for now the sequence mergingn is enough in terms of learning. So we can see what is happening at each step.

# String representation

- Tokens are stored in hex values <66>,<6f>...etc
- For production tokenizers, they use integer values and stores merges as thress numbers -> (token_a,token_b)=new_token.
  ### Using Strings (Token Representation)

| Pros                                                            | Cons                                    |
| :-------------------------------------------------------------- | :-------------------------------------- |
| • Highly debuggable                                             | • Wastes memory / high storage overhead |
| • Explicitly clear (can see exactly what each token represents) | • Requires parsing overhead             |

## Vocabulary management

- For learning :- Keep every merge including off-patterns (outliers).
- Sampling optimization is good enough for learning purposes.
- For production level: Prunes low-frequency merges and sampling optimization is not a good option.
- For plain English text, this approach of encoding-decoding following UTF-8 standards is perfect and optimized as every valid UTF-8 sequence encodes and decodes perfectly.
- However, this approach is less optimized for Non-English characters.

# Why these choices?

- There are libraries like "Tokenizer" and "Tikloren_rs" in rust that are production-grade and addresses the issues that popped up in the learning process.
- The reason for not using them is because the goal here is understanding and learning, not optimizng.
- To implement and to show every step explicitly.
- Can trace exactly what happens during training and decoding.
- Understanding why encoding increases with vocabulary size.
- Understanding token representation of hex values.
- These insights are more valuable than 10x speedup.
- For learning, Feste works perfectly.

# PART 2: TENSOR OPERATIONS

## PURPOSE: To build a library in Rust to power up the Feste Transformers, not for speed optimization but to understand and be clear about what is happening underneath.

- Tensor operation is the mathematical heart of the model.
- It explains the operations happening inside the transformer starting from Attention to Layer Normalization and simplifies to basic computations on multi-dimensional arrays of numbers.
- This part of the article explains how transformers are elegantly constructed systems shaped by how math works on real computers.
