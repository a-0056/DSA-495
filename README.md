# DSA496

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

[Write 3–5 sentences.]

### Q3. Truncation

What content was removed from the long diagnostic message at the artificial
32-token limit? Explain one way that losing this content could affect emotion
classification. Do not claim that truncation caused an observed model error
unless you test that claim.

**Response:**

[Write 2–3 sentences.]

## 2. Specialized encoder classification

### Q4. Baseline and encoder results

Complete the table using the 1,970-message evaluation set.

| Method | Accuracy | Macro-F1 | Inference seconds |
|---|---:|---:|---:|
| Always predict joy | | | N/A |
| DistilBERT emotion classifier | | | |

Which emotion has the lowest DistilBERT recall? Include its recall and support.

**Response:**

[Write 2–3 sentences.]

### Q5. Three encoder errors

Record three incorrect DistilBERT predictions. Include at least one high-score
error if the notebook produces one.

| Example ID | Reference label | Prediction | Model score | Brief observation |
|---|---|---|---:|---|
| | | | | |
| | | | | |
| | | | | |

What pattern, ambiguity, or missing context do you observe? Cite language from
the messages. Remember that a high model score is not proof that the prediction
is correct or that the score is calibrated.

**Response:**

[Write 3–5 sentences.]

## 3. Zero-shot classification

### Q6. Label wording

Complete the development-set comparison.

| Candidate-label formulation | Accuracy | Macro-F1 |
|---|---:|---:|
| A: emotion names | | |
| B: expanded descriptions | | |

Which formulation did the prespecified macro-F1 rule select? Give one example
whose prediction changed when the wording changed. Why is a conclusion based on
only five development messages per class uncertain?

**Response:**

[Write 3–4 sentences.]

### Q7. Final model comparison

Complete the table using the same 1,970 evaluation messages for both models.

| Method | Accuracy | Macro-F1 | Inference seconds |
|---|---:|---:|---:|
| Always predict joy | | | N/A |
| DistilBERT emotion classifier | | | |
| BART zero-shot classifier | | | |

Describe the main performance difference without claiming that this is a
controlled comparison of model architectures.

**Response:**

[Write 2–3 sentences.]

### Q8. Four model disagreements

Record four examples where DistilBERT and BART disagree. Include different
correctness patterns when the notebook makes them available.

| Example ID | Reference | DistilBERT | BART | Who is correct? |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

Choose two of these messages and explain what textual evidence supports each
model’s prediction. If the reference label is debatable, explain why.

**Response:**

[Write 3–5 sentences.]

### Q9. Recommendation and limitations

Which classifier would you use for this fixed six-emotion task? Support your
choice with at least two quantitative results and one finding from error
analysis. Then identify **two limitations** that constrain what the results
establish.

**Response:**

[Write 4–6 sentences.]

## AI-use statement

State whether you used a generative or agentic AI tool for this assignment.

If you used one, briefly identify the tool and explain its role in your work.
Describe what you completed independently, what assistance you incorporated,
and how you reviewed or verified the final result. A concise paragraph is
sufficient; you do not need to include a transcript of your prompts. For
example, you might write:

> I used [tool] to help with [purpose]. I used its suggestions or output for
> [part of the assignment] and reviewed or verified that work by [method].

This example is a guide, not a required format; include the details that best
describe how you used and checked the tool.

If you did not use one, write: “I did not use a generative or agentic AI tool
for this assignment.”
