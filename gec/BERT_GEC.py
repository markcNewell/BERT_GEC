""" 
***Use of BERT Masked Language Model (MLM) for Grammar Error Correction (GEC), without the use of annotated data***

**High level workflow**
 
•	Tokenize the sentence using Spacy

•	Check for spelling errors using Hunspell <- removed

•	For all preposition, determiners & helper verbs, create a set of probable sentences

•	Create a set of sentences with each word “masked”, deleted or an additional determiner, preposition or helper verb added

•	Used BERT Masked Language Model to determine possible suggestions for masks

•	Use the GED model to select appropriate solutions

"""

from typing import List
import torch
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertForMaskedLM
from transformers import pipeline
import requests
from keras_preprocessing.sequence import pad_sequences
import numpy as np
import spacy
from cdifflib import CSequenceMatcher
import logging
from pathlib import Path


# Check to confirm that GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
#torch.cuda.get_device_name(0)

logging.basicConfig(level=logging.INFO)


class GEC_Model:

    det = [
        "the",
        "a",
        "an",
        "this",
        "that",
        "these",
        "those",
        "my",
        "your",
        "his",
        "her",
        "its",
        "our",
        "their",
        "all",
        "both",
        "half",
        "either",
        "neither",
        "each",
        "every",
        "other",
        "another",
        "such",
        "what",
        "rather",
        "quite",
    ]

    # List of common prepositions
    prep = [
        "about",
        "at",
        "by",
        "for",
        "from",
        "in",
        "of",
        "on",
        "to",
        "with",
        "into",
        "during",
        "including",
        "until",
        "against",
        "among",
        "throughout",
        "despite",
        "towards",
        "upon",
        "concerning",
    ]

    # List of helping verbs
    helping_verbs = [
        "am",
        "is",
        "are",
        "was",
        "were",
        "being",
        "been",
        "be",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "can",
        "could",
    ]

    def __init__(self, MLM="mn367/mark-finetuned-imdb") -> None:
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # load previously trained BERT Grammar Error Detection model
        self.modelGED = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2
        )

        if not Path("bert-based-uncased-GED.pth").is_file():
            self.download_file_from_google_drive("1PMzb3VwSUVN2BeVomb3dwjFES9qmlaLe", "bert-based-uncased-GED.pth")

        # restore model
        self.modelGED.load_state_dict(torch.load("bert-based-uncased-GED.pth", map_location=device))
        self.modelGED.eval()

        # Load pre-trained model (weights) for Masked Language Model (MLM)
        self.model_pipeline = pipeline(
            "fill-mask", model=MLM
        )

        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizerLarge = BertTokenizer.from_pretrained("bert-large-uncased")

        self.nlp = spacy.load("en_core_web_sm")
        self.gn = None #TODO: Look at hunspell for windows


    def download_file_from_google_drive(self, id: str, destination: str) -> str:
        print("Trying to fetch {}".format(destination))

        def get_confirm_token(response) -> str:
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    return value

            return None

        def save_response_content(response, destination: str) -> None:
            CHUNK_SIZE = 32768

            with open(destination, "wb") as f:
                for chunk in self.progress_bar(response.iter_content(CHUNK_SIZE)):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)

        URL = "https://docs.google.com/uc?export=download&confirm=t"

        session = requests.Session()

        response = session.get(URL, params = { 'id' : id }, stream = True)
        token = get_confirm_token(response)

        if token:
            params = { 'id' : id, 'confirm' : token }
            response = session.get(URL, params = params, stream = True)

        save_response_content(response, destination)


    def progress_bar(self, some_iter):
        try:
            from tqdm import tqdm
            return tqdm(some_iter)
        except ModuleNotFoundError:
            return some_iter


    def check_GE(self, sents: List[str]) -> tuple:
        """Check of the input sentences have grammatical errors

        :param list: list of sentences
        :return: error, probabilities
        :rtype: (boolean, (float, float))
        """

        # Create sentence) and label lists
        # We need to add special tokens at the beginning and end of each sentence
        # for BERT to work properly
        sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sents]
        labels = [0]

        tokenized_texts = [self.tokenizer.tokenize(sent) for sent in sentences]

        # Padding Sentences
        # Set the maximum sequence length. The longest sequence in our training set
        # is 47, but we'll leave room on the end anyway.
        # In the original paper, the authors used a length of 512.
        MAX_LEN = 128

        predictions = []
        true_labels = []

        # Pad our input tokens
        input_ids = pad_sequences(
            [self.tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
            maxlen=MAX_LEN,
            dtype="long",
            truncating="post",
            padding="post",
        )

        # Index Numbers and Padding
        input_ids = [self.tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

        # pad sentences
        input_ids = pad_sequences(
            input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post"
        )

        # Attention masks
        # Create attention masks
        attention_masks = []

        # Create a mask of 1s for each token followed by 0s for padding
        for seq in input_ids:
            seq_mask = [float(i > 0) for i in seq]
            attention_masks.append(seq_mask)

        prediction_inputs = torch.tensor(input_ids)
        prediction_masks = torch.tensor(attention_masks)
        prediction_labels = torch.tensor(labels)

        with torch.no_grad():
            # Forward pass, calculate logit predictions
            logits = self.modelGED(
                prediction_inputs, token_type_ids=None, attention_mask=prediction_masks
            )

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        # label_ids = b_labels.to("cpu").numpy()

        # Store predictions and true labels
        predictions.append(logits)
        # true_labels.append(label_ids)

        #   print(predictions)
        flat_predictions = [item for sublist in predictions for item in sublist]
        #   print(flat_predictions)
        prob_vals = flat_predictions
        flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
        # flat_true_labels = [item for sublist in true_labels for item in sublist]
        #   print(flat_predictions)
        return flat_predictions, prob_vals


    def create_spelling_set(self, org_text: str, softmax_threshold: float = 0.6) -> List[str]:
        """Create a set of sentences which have possible corrected spellings"""

        sent = org_text
        sent = sent.lower()
        sent = sent.strip().split()

        proc_sent = self.nlp.tokenizer.tokens_from_list(
            sent
        )  # https://github.com/explosion/spaCy/issues/5399
        self.nlp.tagger(proc_sent)

        sentences = []

        for tok in proc_sent:
            # check for spelling for alphanumeric
            if tok.text.isalpha() and not self.gb.spell(tok.text):
                new_sent = sent[:]
                # append new sentences with possible corrections
                for sugg in self.gb.suggest(tok.text):
                    new_sent[tok.i] = sugg
                    sentences.append(" ".join(sent))

        spelling_sentences = sentences

        # retain new sentences which have a
        # minimum chance of correctness using BERT GED
        new_sentences = []

        for sent in spelling_sentences:
            no_error, prob_val = self.check_GE([sent])
            exps = [np.exp(i) for i in prob_val[0]]
            sum_of_exps = sum(exps)
            softmax = [j / sum_of_exps for j in exps]
            if softmax[1] > softmax_threshold:
                new_sentences.append(sent)

        # if no corrections, append the original sentence
        if len(spelling_sentences) == 0:
            spelling_sentences.append(" ".join(sent))

        # eliminate dupllicates
        [spelling_sentences.append(sent) for sent in new_sentences]
        spelling_sentences = list(dict.fromkeys(spelling_sentences))

        return spelling_sentences


    def create_grammar_set(self, spelling_sentences: List[str], softmax_threshold: float = 0.6) -> List[str]:
        """create a new set of sentences with deleted determiners,
        prepositions & helping verbs

        """
        new_sentences = []

        for text in spelling_sentences:
            sent = text.strip().split()
            for i in range(len(sent)):
                new_sent = sent[:]

                if new_sent[i] not in list(set(GEC_Model.det + GEC_Model.prep + GEC_Model.helping_verbs)):
                    continue

                del new_sent[i]
                text = " ".join(new_sent)

                # retain new sentences which have a
                # minimum chance of correctness using BERT GED
                no_error, prob_val = self.check_GE([text])
                exps = [np.exp(i) for i in prob_val[0]]
                sum_of_exps = sum(exps)
                softmax = [j / sum_of_exps for j in exps]
                if softmax[1] > softmax_threshold:
                    new_sentences.append(text)

        # eliminate dupllicates
        [spelling_sentences.append(sent) for sent in new_sentences]
        spelling_sentences = list(dict.fromkeys(spelling_sentences))
        return spelling_sentences


    def create_mask_set(self, spelling_sentences: List[str]) -> List[str]:
        """For each input sentence create 2 sentences
        (1) [MASK] each word
        (2) [MASK] for each space between words
        """
        sentences = []

        for sent in spelling_sentences:
            sent = sent.strip().split()
            for i in range(len(sent)):
                # (1) [MASK] each word
                new_sent = sent[:]
                new_sent[i] = "[MASK]"
                text = " ".join(new_sent)
                new_sent = "[CLS] " + text + " [SEP]"
                sentences.append(new_sent)

                # (2) [MASK] for each space between words
                new_sent = sent[:]
                new_sent.insert(i, "[MASK]")
                text = " ".join(new_sent)
                new_sent = "[CLS] " + text + " [SEP]"
                sentences.append(new_sent)

        return sentences

    def check_grammar(self, org_sent: str, sentences: List[str], spelling_sentences: List[str], softmax_threshold: float = 0.96, similarity_threshold: float = 0.3):
        """check grammar for the input sentences"""

        n = len(sentences)

        # what is the tokenized value of [MASK]. Usually 103
        text = "[MASK]"
        tokenized_text = self.tokenizerLarge.tokenize(text)
        mask_token = self.tokenizerLarge.convert_tokens_to_ids(tokenized_text)[0]

        LM_sentences = []
        new_sentences = []
        l = len(org_sent.strip().split()) * 2  # l is no of sentencees
        mask = False  # flag indicating if we are processing space MASK

        for i, sent in enumerate(sentences):
            # Predict all tokens
            with torch.no_grad():
                predictions = self.model_pipeline(sent)

            # predicted token
            predicted_token = predictions[0]['token_str']
            
            text = sent.strip().split()
            mask_index = text.index("[MASK]")

            if not mask:
                # case of MASKed words
                mask = True
                text[mask_index] = predicted_token

                try:
                    # retrieve original word
                    org_word = (
                        spelling_sentences[i // l].strip().split()[mask_index - 1]
                    )
                except:
                    continue

                # use SequenceMatcher to see if predicted word is similar to original word
                if org_word == predicted_token:
                    continue

                if CSequenceMatcher(None, org_word, predicted_token).ratio() < similarity_threshold:
                    if org_word not in list(
                        set(GEC_Model.det + GEC_Model.prep + GEC_Model.helping_verbs)
                    ) or predicted_token not in list(set(GEC_Model.det + GEC_Model.prep + GEC_Model.helping_verbs)):
                        continue

            else:
                # case for MASKed spaces
                mask = False

                # only allow determiners / prepositions  / helping verbs in spaces
                if predicted_token in list(set(GEC_Model.det + GEC_Model.prep + GEC_Model.helping_verbs)):
                    text[mask_index] = predicted_token
                else:
                    continue

            text.remove("[SEP]")
            text.remove("[CLS]")
            new_sent = " ".join(text)

            # retain new sentences which have a
            # minimum chance of correctness using BERT GED
            no_error, prob_val = self.check_GE([new_sent])

            exps = [np.exp(i) for i in prob_val[0]]
            sum_of_exps = sum(exps)
            softmax = [j / sum_of_exps for j in exps]

            print(new_sent, softmax)

            if no_error and softmax[1] > softmax_threshold:
                new_sentences.append(new_sent)

        # remove duplicate suggestions
        spelling_sentences = []
        [spelling_sentences.append(sent) for sent in new_sentences]
        spelling_sentences = list(dict.fromkeys(spelling_sentences))
        spelling_sentences

        return spelling_sentences


if __name__ == "__main__":
    pred = ["obviously the tire now is the prior limitation"]

    model = GEC_Model()

    for sent in pred:

        print("Input Sentence >>> " + sent)

        sentences = [sent] #model.create_spelling_set(sent)
        spelling_sentences = model.create_grammar_set(sentences)
        sentences = model.create_mask_set(spelling_sentences)

        print("processing {0} possibilities".format(len(sentences)))

        sentences = model.check_grammar(sent, sentences, spelling_sentences)

        print("Suggestions & Probabilities")

        if len(sentences) == 0:
            print("None")
            continue

        no_error, prob_val = model.check_GE(sentences)

        for i in range(len(prob_val)):
            exps = [np.exp(i) for i in prob_val[i]]
            sum_of_exps = sum(exps)
            softmax = [j / sum_of_exps for j in exps]
            print("{0} - {1:0.4f}%".format(sentences[i], softmax[1] * 100))

        print("-" * 60)
        print()
