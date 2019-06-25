import os
import json
import tensorflow as tf
import byte_pair_encodings as bpe
import read_data
import model

class InputExample(object):

  def __init__(self, guid, text):
    self.guid = guid
    self.text = text

class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size."""

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               label_ids,
               label_mask,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.label_ids = label_ids
    self.label_mask = label_mask
    self.is_real_example = is_real_example

class InputProcessing():

    def __init__(self):
        self.model_dir = '.\\gpt2\\models'
        self.model_name = '117M'
        self.hyperparams = model.default_hparams()

    def corpus2ids(self, corpus):
        encoder, bpe_merges = bpe.load_vocab_and_encoder(self.model_dir, self.model_name)
        bpe_encoder = bpe.Encoder(encoder=encoder, bpe_merges=bpe_merges)
        ids = []
        for doc in corpus:
            ids.append(bpe_encoder.encode(doc))
        return ids

    def id2corpus(self, ids):
        encoder, bpe_merges = bpe.load_vocab_and_encoder(self.model_dir, self.model_name)
        bpe_encoder = bpe.Encoder(encoder=encoder, bpe_merges=bpe_merges)
        corpus = []
        for doc in ids:
            corpus.append(bpe_encoder.decode(doc))
        return corpus

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
      """Truncates a sequence pair in place to the maximum length."""

      # This is a simple heuristic which will always truncate the longer sequence
      # one token at a time. This makes more sense than truncating an equal percent
      # of tokens from each, since if one sequence is very short then each token
      # that's truncated likely contains more information than a longer sequence.
      while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
          break
        if len(tokens_a) > len(tokens_b):
          tokens_a.pop()
        else:
          tokens_b.pop()

    def convert_single_example(self, ex_index, example, max_seq_length):
      """Converts a single `InputExample` into a single `InputFeatures`."""

      if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            label_ids=[0] * max_seq_length,
            is_real_example=False)

      tokens = example.text
      if len(tokens) > max_seq_length - 1:
        tokens_i = tokens[0:max_seq_length-1]
        tokens_o = tokens[0:max_seq_length-1]
      else:
        tokens_i = tokens[0:len(tokens)-1]
        tokens_o = tokens[0:len(tokens)-1]

      input_ids = []
      input_ids.append(50257) # "START"
      for token in tokens_i:
        input_ids.append(token)

      label_ids = []
      for token in tokens_o:
        label_ids.append(token)
      label_ids.append(50256) # "<|endoftext|>"

      # The mask has 1 for real tokens and 0 for padding tokens. Only real
      # tokens are attended to.
      input_mask = [1] * len(input_ids)
      label_mask = [1] * len(label_ids)

      # Zero-pad up to the sequence length.
      while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        label_ids.append(0)
        label_mask.append(0)

      assert len(input_ids) == max_seq_length
      assert len(input_mask) == max_seq_length
      assert len(label_ids) == max_seq_length
      assert len(label_mask) == max_seq_length

      if ex_index < 5:
        print("*** Example ***")
        print("guid: %s" % (example.guid))
        print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        print("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        print("label_mask: %s" % " ".join([str(x) for x in label_mask]))

      feature = InputFeatures(
          input_ids=input_ids,
          input_mask=input_mask,
          label_ids=label_ids,
          label_mask = label_mask,
          is_real_example=True)
      return feature

    def convert_examples_to_features(self, examples, max_seq_length):
      """Convert a set of `InputExample`s to a list of `InputFeatures`."""

      features = []
      for (ex_index, example) in enumerate(examples):
        if ex_index % 10 == 0:
          tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature = self.convert_single_example(ex_index, example, max_seq_length)
        features.append(feature)
      return features

    def _inputExampleIter(self, data):
        for i, d in enumerate(data):
            yield InputExample(guid=i, text=d)

    def make_input_features(self, data):
        input_examples = [i for i in self._inputExampleIter(data)]
        features = self.convert_examples_to_features(examples=input_examples,
                                                     max_seq_length=self.hyperparams.n_ctx)
        return features


def main():
    tr = InputProcessing()
    corpus = read_data.corpus()
    corpus_ids = tr.corpus2ids(corpus)
    #corpus_ = id2corpus(corpus_ids)
    features = tr.make_input_features(corpus_ids[:20])


if __name__ == '__main__':
    main()
