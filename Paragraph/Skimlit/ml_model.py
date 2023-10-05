import pandas as pd
import numpy as np
import tensorflow as tf
#from tensorflow.keras import layers
#import tensorflow_text as text
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from spacy.lang.en import English
# from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
# from tensorflow.keras.preprocessing.text import Tokenizer
import cv2
import matplotlib.pyplot as plt

def get_lines(filename):

  with open(filename,'r') as f:
    return f.readlines()

data_dir = 'C:/Users/anany/Paragraph/pubmed-rct/PubMed_20k_RCT_numbers_replaced_with_at_sign/'  

train_lines = get_lines(data_dir + 'train.txt')


def preprocess_text_with_line_numbers(filename):

  input_lines = get_lines(filename) # get all lines from filename
  empty_abstract = "" # create an empty abstract
  sample_abstracts = [] # create an empty list of abstracts
  
  # Loop through each line in target file
  for line in input_lines:
    if line.startswith("###"): # check to see if line is an ID line
      abstract_id = line
      empty_abstract = "" # reset abstract string
    elif line.isspace(): # check to see if line is a new line
      abstract_line_split = empty_abstract.splitlines() # split abstract into separate lines

      # Iterate through each line in abstract and count them at the same time
      for abstract_line_number, abstract_line in enumerate(abstract_line_split):
        line_data = {} # create empty dict to store data from line
        target_text_split = abstract_line.split("\t") # split target label from text
        line_data["target"] = target_text_split[0] # get target label
        line_data["text"] = target_text_split[1].lower() # get target text and lower it
        line_data["line_number"] = abstract_line_number # what number line does the line appear in the abstract?
        line_data["total_lines"] = len(abstract_line_split) - 1 # how many total lines are in the abstract? (start from 0)
        sample_abstracts.append(line_data) # add line data to abstract samples list
    
    else: # if the above conditions aren't fulfilled, the line contains a labelled sentence
      empty_abstract += line
  return sample_abstracts


train_samples = preprocess_text_with_line_numbers(data_dir + 'train.txt')
val_samples = preprocess_text_with_line_numbers(data_dir + 'dev.txt')
test_samples = preprocess_text_with_line_numbers(data_dir + 'test.txt')


train_df = pd.DataFrame(train_samples)
val_df = pd.DataFrame(val_samples)
test_df = pd.DataFrame(test_samples)


train_sentences = train_df['text'].tolist()
val_sentences = val_df['text'].tolist()
test_sentences = test_df['text'].tolist()


one_hot_encoder = OneHotEncoder(sparse=False)
train_labels_one_hot = one_hot_encoder.fit_transform(train_df["target"].to_numpy().reshape(-1, 1))
val_labels_one_hot = one_hot_encoder.transform(val_df["target"].to_numpy().reshape(-1, 1))
test_labels_one_hot = one_hot_encoder.transform(test_df["target"].to_numpy().reshape(-1, 1))


label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_df["target"].to_numpy())
val_labels_encoded = label_encoder.transform(val_df["target"].to_numpy())
test_labels_encoded = label_encoder.transform(test_df["target"].to_numpy())


sentence_lengths = [len(sentence.split()) for sentence in train_sentences]
avg_sent_len = np.mean(sentence_lengths)

output_seq_len = int(np.percentile(sentence_lengths , 95))

max_tokens = 68000
#text_vectorizer = TextVectorization(max_tokens = max_tokens,
#                                    output_sequence_length = output_seq_len)
#text_vectorizer.adapt(train_sentences)

#rct_20k_text_vocab = text_vectorizer.get_vocabulary()

# Assuming you have a list of texts called 'corpus' containing your training data

# # Create a tokenizer
# tokenizer = Tokenizer()
# # Fit the tokenizer on the corpus
# tokenizer.fit_on_texts(train_sentences)
# token_embed = tf.keras.layers.Embedding(input_dim=len(rct_20k_text_vocab), # length of vocabulary
#                                output_dim=128, # Note: different embedding sizes result in drastically different numbers of parameters to train
#                                # Use masking to handle variable sequence lengths (save space)
#                                mask_zero=True,
#                                name="token_embedding") 


train_dataset = tf.data.Dataset.from_tensor_slices((train_sentences, train_labels_one_hot))
valid_dataset = tf.data.Dataset.from_tensor_slices((val_sentences, val_labels_one_hot))
test_dataset = tf.data.Dataset.from_tensor_slices((test_sentences, test_labels_one_hot))


train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
valid_dataset = valid_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

num_classes = len(label_encoder.classes_)
class_names = label_encoder.classes_
num_classes, class_names

# Make function to split sentences into characters
def split_chars(text):
  return " ".join(list(text))


from spacy.lang.en import English

def make_prediction(model, text):
    nlp = English()
    #sentencizer = nlp.add_pipe('sentencizer')
    nlp.add_pipe('sentencizer')
    doc = nlp(text)
    abstract_lines = [str(sent) for sent in list(doc.sents)]

    total_lines_in_sample = len(abstract_lines)

    sample_lines = []
    for i, line in enumerate(abstract_lines):
        sample_dict = {}
        sample_dict["text"] = str(line)
        sample_dict["line_number"] = i
        sample_dict["total_lines"] = total_lines_in_sample - 1
        sample_lines.append(sample_dict)

    test_abstract_line_numbers = [line["line_number"] for line in sample_lines]
    test_abstract_line_numbers_one_hot = tf.one_hot(test_abstract_line_numbers, depth=15)

    test_abstract_total_lines = [line["total_lines"] for line in sample_lines]
    test_abstract_total_lines_one_hot = tf.one_hot(test_abstract_total_lines, depth=20)

    abstract_chars = [split_chars(sentence) for sentence in abstract_lines]  # Adjust this as per your needs

    test_abstract_pred_probs = model.predict(abstract_lines)

    test_abstract_preds = tf.argmax(test_abstract_pred_probs, axis=1)
    test_abstract_pred_classes = [label_encoder.classes_[i] for i in test_abstract_preds]
    #test_abstract_pred_classes =  "\u001b[0m" + test_abstract_pred_classes + "\u001b[0m"
   # for i in test_abstract_pred_classes:
    #   i = "\u001b[0m" + i + "\u001b[0m"
    # predictions = []
    # for i, line in enumerate(abstract_lines):
    #     predictions.append(f"{test_abstract_pred_classes[i]} : {line}")

    # grp_preds = {}
    # for element in predictions:
    #    key, value = element.split(':')
    #    if key not in grp_preds:
    #      grp_preds[key] = []
    #    grp_preds[key].append(value)
    predictions = {}
    for i, line in enumerate(abstract_lines):
        predictions[line] = test_abstract_pred_classes[i]

    grouped_predictions = {}
    for key, value in predictions.items():
        if value not in grouped_predictions:
            grouped_predictions[value] = []
        grouped_predictions[value].append(key)

    # for target, lines in grouped_predictions.items():
    #     print(f"Target: {target}")
    #     for line in lines:
    #         print(line)
    #     print('\n')

    #grp_preds = [(item['background'],item['objective'],item['method'],item['result'],item['conclusion']) for item in predictions]
    return grouped_predictions



