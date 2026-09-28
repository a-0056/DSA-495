# DSA495

# Lab 1: Emotion Classification and Error Analysis

**Student:** Abigail Ma

Write concise answers supported by your executed notebook. Tables can contain
exact results; your prose should interpret them rather than repeat every number.
Aim for about **400–800 words total**, excluding tables, file descriptions, and
the AI-use statement. One short paragraph per question is enough.

## Purpose

In 2–3 sentences, describe the classification task and what the comparison among
the constant baseline, specialized DistilBERT classifier, and BART zero-shot
classifier is intended to show.

**Response:**

The classification task is used to classify the six emotions expressed in each of the
responses, including the emotions sadness, joy, love, anger, fear, and surprise. The
constant baseline is the five sample data points taken from each of the categories of
emotions (30 points total). These points are set aside for development, and the DistilBERT
and BART zero-shot classifiers try to predict the emotions expressed for the rest of
the data points not taken as a sample.

## Files and rerun instructions

List every file included in your submission and briefly describe it.

- `Lab1.ipynb` (or .py): This lab contains the code that I analyzed to answer the questions
      in the "README.md" document.
- `README.md`: This contains all responses and is the reproducibility document.
- [Add any other submitted files if any, or write “No additional files.”]

To run the analysis:

1. Open the notebook in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Confirm that the course data are available in Google Drive at
   `DSA495-2026/Labs/Lab 1`.
4. Run all notebook cells from top to bottom.
5. [Add any additional instruction needed to reproduce your submission.]

The notebook contains two designated student code blocks and one model-selection
line. Complete those sections yourself; the surrounding setup and model-inference
code is supplied.

## 1. Data and tokenization

### Q1. Development and evaluation data

Report the number of development and evaluation messages. Which evaluation
class is least common, and why does that make accuracy alone insufficient?

**Response:**

There are 30 development messages in total taken as a sample, with 5 for each emotion.
The rest of the data points are evaluated as shown in the evaluation column. The total number
of these evaluation data points is 1970. The least common evaluation class is surprise,
and this makes accuracy alone insufficient because surprise only makes up a small percentage
of all the data points, while other emotions such as joy and sadness have a significantly
larger number of data points for that emotion. This may suggest bias in the model.

### Q2. What do the tokenizers receive?

Using specific rows from your token table, explain **two meaningful differences**
between the DistilBERT and BART tokenizers. At least one explanation must refer
to one of your two student-written examples.

**Response:**

The BART tokenizer starts some of the tokens with a capital "Ĝ" while the DistilBERT
tokenizer does not add any values to the beginning of each token. This can be seen with
the example "I1", where many of the tokens displayed in token_strings for the BART tokenizer 
start with a capital "Ĝ" and the DistilBERT tokenizer's tokens do not start with any letter
or number. The BART tokenizer also maintains tokens starting with a capitalized letter, while 
the DistilBERT tokenizer converts all capitalized tokens into lowercase tokens. This can be seen 
in the example "S1", where the word "Dolphin" is converted to a lowercase token by the DistilBERT 
tokenizer. Meanwhile, the BART tokenizer keeps the token capitalized.

### Q3. Truncation

What content was removed from the long diagnostic message at the artificial
32-token limit? Explain one way that losing this content could affect emotion
classification. Do not claim that truncation caused an observed model error
unless you test that claim.

**Response:**

All content after approximately the second sentence was removed from the long
diagnostic message at the artificial 32-token limit. This includes the last sentence,
which changes the emotion of the sentence to fear instead because it contains the
phrase "I am terrified about what happens tomorrow." which is essential for identifying
the emotion of fear. Losing this content will affect the classification because the model
is only analyzing the part where the sentence says "I described the train ride, the weather, 
and every stop along the way.", which does not convey the entire message of emotion of
fear as expressed in the last sentence.

## 2. Specialized encoder classification

### Q4. Baseline and encoder results

Complete the table using the 1,970-message evaluation set.

| Method | Accuracy | Macro-F1 | Inference seconds |
|---|---:|---:|---:|
| Always predict joy | 0.350254 | 0.86466 | N/A |
| DistilBERT emotion classifier | 0.924365 | 0.880256 | 4.478255 |

Which emotion has the lowest DistilBERT recall? Include its recall and support.

**Response:**

The emotion that has the lowest DistilBERT recall is surprise, with the recall
value being 0.754098 and the support being 61.0.

### Q5. Three encoder errors

Record three incorrect DistilBERT predictions. Include at least one high-score
error if the notebook produces one.

| Example ID | Reference label | Prediction | Model score | Brief observation |
|---|---|---|---:|---|
| emotion_test01314 | ive blogged and i feel strange about it | surprise | fear | 0.998749852180481 |
| emotion_test_01377 | i walked to school he felt the bounce in his step the overjoyed feelings of youth and the thrill of excitement of coming to school and meeting his beloved friends | love | joy | 0.9985522627830505 |
| emotion_test00816 | whenever i put myself in other shoes and try to make the person happy | anger | joy | 0.9984239339828491 |

What pattern, ambiguity, or missing context do you observe? Cite language from
the messages. Remember that a high model score is not proof that the prediction
is correct or that the score is calibrated.

**Response:**

I noticed that the emotions of joy and love are hard to distinguish clearly by the model
because in emotion_test_01377, the model predicted that the emotion was joy when it was love, based
on words like "overjoyed" and phrases like "the thrill of excitement". This in turn led to the model
having a high encoder_score (certainty). I also saw that the pattern of the model over-relied on
certain words to make its prediction. For example, in emotion_test_00816, the word "happy" is present
in the text, and the model predicted that the text was expressing joy mainly because of this word.

## 3. Zero-shot classification

### Q6. Label wording

Complete the development-set comparison.

| Candidate-label formulation | Accuracy | Macro-F1 |
|---|---:|---:|
| A: emotion names | 0.500000 | 0.464388 |
| B: expanded descriptions | 0.566667 | 0.552279 |

Which formulation did the prespecified macro-F1 rule select? Give one example
whose prediction changed when the wording changed. Why is a conclusion based on
only five development messages per class uncertain?

**Response:**

The prespecified macro-F1 rule selected the expanded descriptions formulation.
An example whose prediction changed when the wording changed is emotion_test_00332,
where the label is sadness, but the model's initial prediction was surprise. After
using the expanded descriptions formulation, however, the model predicted that the
emotion being expressed was sadness instead, which matches the correct label. A
conclusion based on only five development messages per class is uncertain because
five development messages are not enough for the model to analyze the full spectrum of
expressions for one emotion.

### Q7. Final model comparison

Complete the table using the same 1,970 evaluation messages for both models.

| Method | Accuracy | Macro-F1 | Inference seconds |
|---|---:|---:|---:|
| Always predict joy | 0.350254 | 0.086466 | N/A |
| DistilBERT emotion classifier | 0.924365 | 0.880256 | 171.198248 |
| BART zero-shot classifier | 0.536548 | 0.479474 | 7289.840730 |

Describe the main performance difference without claiming that this is a
controlled comparison of model architectures.

**Response:**

The main performance difference between the DistilBERT and BART zero-shot models
is that DistilBERT performed better and was more accurate when identifying
what emotions are being expressed, with an accuracy rate of 0.924365 compared to the
BART zero-shot's accuracy rate of 0.536548. The time used to run the model 
171.198248 seconds, was significantly less than the BART zero-shot model, which 
took 7289.840730 seconds to run.

### Q8. Four model disagreements

Record four examples where DistilBERT and BART disagree. Include different
correctness patterns when the notebook makes them available.

| Example ID | Reference | DistilBERT | BART | Who is correct? |
|---|---|---|---|---|
emotion_test_00002 | i never make her separate from me because i don t ever want her 
to feel like i m ashamed with her| sadness | sadness | love | encoder only correct |
emotion_test_00072 | i am right handed however i play billiards left handed naturally 
so me trying to play right handed feels weird | surprise | fear | surprise | zero-shot only correct |
emotion_test_00098 | i feel my heart is tortured by what i have done | anger | fear | sadness | both incorrect |
emotion_test_00004 | i was feeling a little vain when i did this one | sadness | sadness | surprise | encoder only correct |

Choose two of these messages and explain what textual evidence supports each
model’s prediction. If the reference label is debatable, explain why.

**Response:**

For emotion_test_0002, the DistilBERT model determined that the emotion being expressed
was sadness, while the BART zero-shot model identified the emotion as love. The DistilBERT
model likely relied on the word "ashamed" to identify the emotion of sadness, while the
BART zero-shot model focused on the phrase "i never make her separate from me" to identify
love.

For emotion_test_00072, the DistilBERT model determined that the text was expressing fear
while the BART zero-shot model found that the emotion in the text is surprise. The DistilBERT
identified fear in the text through the words "however" and "weird" since these words suggested
fear, while the BART zero-shot model identified surprise through the phrase "right handed
however i play billiards left handed naturally" which infers the emotion of surprise.

### Q9. Recommendation and limitations

Which classifier would you use for this fixed six-emotion task? Support your
choice with at least two quantitative results and one finding from error
analysis. Then identify **two limitations** that constrain what the results
establish.

**Response:**

I would use the DistilBERT model because it demonstrates a higher overall accuracy rate than
the BART zero-shot model, and the DistilBERT model had a higher macro-F1 score compared to the
BART zero-shot model's macro-F1 score, which means that the DistilBERT model predicted better
across all categories of emotion. To start, the DistilBERT model had an overall accuracy 
rate of 0.924365, while the BART zero-shot model had an overall accuracy rate of 0.536548. The
DistilBERT model also had a macro-F1 score of 0.880256, and the BART zero-shot model had a 
macro-F1 score of 0.552279. One of the limitations that are present, however, is that the DistilBERT
model had prior training using content from the dataset, while BART did not have any training from
the dataset at all. Additionally, the scores for DistilBERT and BART are not comparable because
they went through different classification procedures, so their scores are not directly comparable.

## AI-use statement

I did not use a generative or agentic AI tool for this assignment.
