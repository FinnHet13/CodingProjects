# Sentiment Analysis Bachelor Thesis

## Overview
As part of my bachelor thesis, I wanted to find a pre-trained transformer model to perform text-based sentiment analysis on a dataset of customer call.
I evaluated three pre-trained transformer models from Hugging Face on their sentiment analysis performance on the EmoWoz dataset.
The SieBERT model performed best with an accuracy of 91.7%.

## Objectives
- Evaluate which of the three models has the highest accuracy
- Write efficient code to minimize compute time for analyzing sentiment

## Dataset
The dataset used for benchmarking is the publicly available EmoWOZ dataset (Feng et al., 2022). 
The EmoWOZ dataset consists of transcripts of task-oriented dialogues where a customer and an agent “converse […] to complete tasks determined by [customer] goals” (Feng et al., 2022, pg.1). 
The dataset contains 10,438 multi-conversation-turn agent-customer dialogues taken from the MultiWOZ dataset (Budzianowski et al., 2020) 
and 996 chatbot-customer dialogues that each have a human-labelled sentiment score positive, negative or neutral for each conversation turn of the customer. 
Only MultiWOZ dialogues were considered. 
Additionally, all conversation turns labelled neutral were removed, leaving 20,219 conversation turns with negative or positive sentiment. 
This subset of EmoWOZ was used to benchmark the different models.

EmoWOZ was chosen as the benchmarking dataset for two reasons. 
First, to the best of my knowledge, none of the publicly available text-based transfer learning models have been trained on EmoWOZ. 
Therefore, the benchmark is fair as no model has the advantage of having already seen any part of the dataset. 
Second, EmoWOZ is the publicly available sentiment analysis dataset that best represents actual customer calls. 
It best represents actual customer calls as these too are task-oriented dialogues where a customer wants to achieve a goal by conversing with a call agent. 
Therefore, whichever model performs best on EmoWOZ should also perform best on other customer call datasets.

## Methodology
- Dataset cleaning: [1_EmoWoz_Setup.ipynb](1_EmoWoz_Setup.ipynb)
- Text pre-processing (text cleaning and stopword removal): [3_Sentiment_analysis.ipynb](3_Sentiment_analysis.ipynb) and [3_Sentiment_analysis_multithread_batch_processing.ipynb](3_Sentiment_analysis_multithread_batch_processing.ipynb)
- Sentiment analysis
  - Sequential processing of text with three models: [3_Sentiment_analysis.ipynb](3_Sentiment_analysis.ipynb)
  - Parallel processing of text with three models with batching: [3_Sentiment_analysis_multithread_batch_processing.ipynb](3_Sentiment_analysis_multithread_batch_processing.ipynb)
- Model evaluation [3_Sentiment_analysis.ipynb](3_Sentiment_analysis.ipynb) and [3_Sentiment_analysis_multithread_batch_processing.ipynb](3_Sentiment_analysis_multithread_batch_processing.ipynb)

## Results
Comparison of accuracy of three models:
| Model   | Accuracy |
|---------|----------|
| SieBERT | 0.917058 |
| RoBERTa | 0.885256 |
| XLNet   | 0.864533 |

Comparison of time taken for three models to complete sentiment analysis with different processing methods:
| Processing method used              | Time taken |
|-------------------------------------|------------|
| Parallel processing with batching   | 1:03 hr    |
| Sequential processing               | 1:45 hr    |

## Dependencies
See [dependencies.yaml](dependencies.yaml)

## References
- Feng, S., Lubis, N., Geishauser, C., Lin, H.-C., Heck, M., Van Niekerk, C., & Gašić, M. (2022). EmoWOZ: A Large-Scale Corpus and Labelling Scheme for Emotion Recognition in Task-Oriented Dialogue Systems. https://www.researchgate.net/publication/354542307_EmoWOZ_A_Large-Scale_Corpus_and_Labelling_Scheme_for_Emotion_in_Task-Oriented_Dialogue_Systems
- Budzianowski, P., Wen, T.-H., Tseng, B.-H., Casanueva, I., Ultes, S., Ramadan, O., & Gašić, M. (2020). MultiWOZ -A Large-Scale Multi-Domain Wizard-of-Oz Dataset for Task-Oriented Dialogue Modelling. https://arxiv.org/pdf/1810.00278.pdf

## Contact
Finn Hetzler

finn.he@protonmail.com
