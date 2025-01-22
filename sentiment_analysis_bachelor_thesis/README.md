## EmoWOZ

This is the dataset for [EmoWOZ: A Large-Scale Corpus and Labelling Scheme for Emotion Recognition in Task-Oriented Dialogue Systems](https://arxiv.org/abs/2109.04919). 


### Data

EmoWOZ contains manually annotated **user emotion** in task-oriented dialogues. It contains dialogues from the following two sources:
* The entire [MultiWOZ](https://github.com/budzianowski/multiwoz) (10438 dialogues)
* Dialogues between human-beings and a machine-generated policy, known as DialMAGE (996 dialogues)

EmoWOZ adopts the same format as MultiWOZ logs with an additional `emotion` field in each log item. The emotion field for each **user turn** contains an array of four items (empty array for system turns). The first three are annotations by three annotators, each identified by an anonymous 8-character global annotator id. The last item contains the `emotion` label obtained either from majority voting or manual resolution, as well as its mapped `sentiment` label.

All DialMAGE dialogues have a dialogue id in the form of ''DMAGExxx.json'' where xxx is a number. For DialMAGE dialogues, we provide `dialog_act` and `span_info` used for response generation in each **system turn**.

The definition for each label is defined as below:
| Emotion Label | Emotion Tokens               | Valence  | Elicitor   | Conduct  | Sentiment Label | Sentiment |
|---------------|------------------------------|----------|------------|----------|-----------------|-----------|
| 0             | Neutral                      | Neutral  | Any        | Polite   | 0               | Neutral   |
| 1             | Fearful, sad, disappointed   | Negative | Event/fact | Polite   | 1               | Negative  |
| 2             | Dissatisfied, disliking      | Negative | Operator   | Polite   | 1               | Negative  |
| 3             | Apologetic                   | Negative | User       | Polite   | 1               | Negative  |
| 4             | Abusive                      | Negative | Operator   | Impolite | 1               | Negative  |
| 5             | Excited, happy, anticipating | Positive | Event/fact | Polite   | 2               | Positive  |
| 6             | Satisfied, liking            | Positive | Operator   | Polite   | 2               | Positive  |

EmoWOZ dataset is licensed under Creative Commons Attribution-NonCommercial 4.0 International Public License and later.


### Citation

If you use EmoWOZ in your own work, please cite our work as follows:

```
@inproceedings{feng-etal-2022-emowoz,
      title={EmoWOZ: A Large-Scale Corpus and Labelling Scheme for Emotion Recognition in Task-Oriented Dialogue Systems}, 
      author={Shutong Feng and Nurul Lubis and Christian Geishauser and Hsien-chin Lin and Michael Heck and Carel van Niekerk and Milica Gašić},
      booktitle = "Proceedings of the 13th Language Resources and Evaluation Conference",
      month = June,
      year = "2022",
      address = "Marseille, France",
      publisher = "European Language Resources Association",
      language = "English",
      ISBN = "979-10-95546-72-6",
}
```
Please note that this dataset should only be used for research purpose.


### Contact

Any questions or bug reports can be sent to shutong.feng@hhu.de