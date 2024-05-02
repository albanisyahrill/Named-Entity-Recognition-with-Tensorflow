# Named Entity Recognition with TensorFlow
## Description

This project use my own architecture model that has been trained to extract and classify text. The model is built using a dataset consisting of various classes. The classes itself are :
1. 'O' (Outside): This is a label given to tokens that do not belong to any entity. In other words, they are not relevant to any named entity in the sentence.
2. 'B-PER' (Beginning of a Person): This label indicates that the token is the beginning of a named entity that is the name of a person. For example, "John" in the sentence would be labelled 'B-PER' if it is the beginning of a person name.
3. 'I-PER' (Inside of a Person): This is the label given to tokens inside the named entity that are Person names, after the initial token that is already labelled 'B-PER'.
4. 'B-ORG' (Beginning of an Organisation): This label indicates that the token is the beginning of a named entity that is the name of an organisation (Organization).
5. 'I-ORG' (Inside of an Organisation): This is the label given to tokens inside the named entity that is the name of the organisation, after the initial token that has been labelled 'B-ORG'.
6. 'B-LOC' (Beginning of a Location): This label indicates that the token is the beginning of a named entity that is the name of a location.
7. 'I-LOC' (Inside of a Location): This is the label given to tokens inside the named entity that is the name of the location, after the initial token that has been labelled 'B-LOC'.
8. 'B-MISC' (Beginning of a Miscellaneous Entity): This label indicates that the token is the beginning of a named entity that is an entity other than a person, organisation or location (Miscellaneous).
9. 'I-MISC' (Inside of a Miscellaneous Entity): This is the label given to tokens inside a named entity that is an entity other than a person, organisation, or location (Miscellaneous), after the initial token that is already labelled 'B-MISC'.

## How it works

1. **Pre-processing of Data**:
    - Parsing nested list to list : parses a nested list into a regular list, and converts it into a numpy array. This function is created so that it can feed datasets into TextVectorization layers.
    - Text Vectorization : handles the tokenization and vectorization in one step and can be easily integrated into your model as a layer.
    - Padded labels : provide padding based on the longest sentence so that it has a consistent sentence length and can be fed to the model during the training process and the padding is -1 (Because labels on conll2003 dataset already in integer value, we skip the Text Vectorizaion step).
    - Generate dataset : turn sentences and labels into a Tensorflow dataset.

2. **Model Building**:
    - Sequential Model.
    - Embedding layer turns positive integers (indexes) into dense vectors of fixed size.
    - Bidirectional LSTM layer to understand the sequential context in the text forward and backward.
    - Dense layer with log_softmax activation function for multi-class classification.

3. **Model Training**:
    - Using Adam's method for optimisation with learning rate = 0.01.
    - Use custom SparseCategoricalCrossentropy loss to ignore value -1 when calculating the loss.
    - Use custom masked accuracy that results an accuracy value that calculates only predictions relevant to the un-masked label (-1 is ignored).
    - Tracked loss and accuracy metrics during training.

4. **Model Evaluation**:
    - Plotting accuracy and loss metrics during training.
    - Measuring accuracy and loss on testing dataset.
    - Feed new text to see the prediction output of the trained model.

## Dataset

The dataset used in this project is a conll2003 dataset from Huggingface. You can also use the same dataset with syntax below :
```bash
from datasets import load_dataset

dataset = load_dataset("conll2003")
```

## Usage
Clone the repository:
```bash
git clone https://github.com/albanisyahrill/Named-Entity-Recognition-with-Tensorflow.git
```
