import os, json
from datetime import datetime

import tensorflow as tf

import byte_pair_encodings as bpe
import read_data, model
from input_processing import InputProcessing


class TrainGPT():

    def __init__(self, lr, max_seq_len,
                 batch_size, num_train_epochs, num_warmup_steps, model_dir):
        self.lr = lr
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.num_train_epochs = num_train_epochs
        self.num_warmup_steps = num_warmup_steps
        self.model_dir = model_dir
        self.hyperparams = model.default_hparams()

    def load_features(self, idx_start=0, idx_end=10000):
        ip = InputProcessing()
        corpus = read_data.corpus()
        corpus_ids = ip.corpus2ids(corpus)
        #corpus_ = id2corpus(corpus_ids)
        features = ip.make_input_features(corpus_ids[idx_start:idx_end])
        return features

    def create_model(self, is_predicting, input_ids, input_mask, labels_ids, label_mask):
        # compute loss
        result = model.model(hparams=self.hyperparams,
                             X=input_ids,
                             input_mask=input_mask)

        logits = result['logits']

        ce_loss = tf.contrib.seq2seq.sequence_loss(logits=logits,
                                                   targets=labels_ids,
                                                   weights=tf.cast(label_mask, tf.float32))

        predicted_words = tf.squeeze(tf.argmax(logits, axis=-1, output_type=tf.int32))
        if is_predicting:
            return predicted_words
        return (ce_loss, predicted_words)


    def model_fn_builder(self, learning_rate, num_train_steps, num_warmup_steps):

      def model_fn(features, labels, mode, params):

        input_ids = features['input_ids']
        input_mask = features['input_mask']
        label_ids = features['label_ids']
        label_mask = features['label_mask']

        is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)
        # TRAIN and EVAL
        if not is_predicting:
          (ce_loss, predicted_labels) = self.create_model(
                is_predicting, input_ids, input_mask, label_ids, label_mask)
          optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate
            )
          global_step = tf.train.get_global_step()
          train_op = optimizer.minimize(
            loss = ce_loss,
            global_step = global_step)

          if mode == tf.estimator.ModeKeys.TRAIN:
            logging_hook = tf.train.LoggingTensorHook({"ce_loss": ce_loss}, every_n_iter=100)
            return tf.estimator.EstimatorSpec(mode=mode,
              loss=ce_loss,
              training_hooks=[logging_hook],
              train_op=train_op)
          else:
            return tf.estimator.EstimatorSpec(mode=mode, loss=ce_loss)

        else:
          predicted_words = self.create_model(
                is_predicting, input_ids, input_mask, labels_ids, label_mask)
          predictions = {'predicted_words': predicted_words}
          return tf.estimator.EstimatorSpec(mode, predictions=predictions)

      return model_fn

    def input_fn_builder(self, features, max_seq_len, batch_size, is_training):
      """Creates an `input_fn` closure to be passed to TPUEstimator."""

      all_input_ids = []
      all_input_mask = []
      all_label_ids = []
      all_label_mask = []

      for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_label_ids.append(feature.label_ids)
        all_label_mask.append(feature.label_mask)

      def input_fn(params):
        """The actual input function."""
        batch_size = self.batch_size

        num_examples = len(features)

        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, max_seq_len],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, max_seq_len],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, max_seq_len],
                    dtype=tf.int32),
            "label_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, max_seq_len],
                    dtype=tf.int32),
        })

        if is_training:
          d = d.repeat()
          d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size)
        return d

      return input_fn

    def train(self, train_features, training=True):
        num_train_steps = int(len(train_features) / self.batch_size * self.num_train_epochs)
        num_warmup_steps = int(num_train_steps * self.num_warmup_steps)
        self.steps_in_single_epoch = int(len(train_features) / self.batch_size)

        run_config = tf.estimator.RunConfig(
            model_dir = self.model_dir,
            save_checkpoints_steps = self.steps_in_single_epoch,
            keep_checkpoint_max = 5
        )

        model_fn = self.model_fn_builder(
          learning_rate = self.lr,
          num_train_steps = num_train_steps,
          num_warmup_steps = num_warmup_steps
        )

        self.estimator = tf.estimator.Estimator(
          model_fn=model_fn,
          config=run_config,
          params={
              "batch_size": self.batch_size
          }
        )

        self.train_input_fn = self.input_fn_builder(
            features = train_features,
            max_seq_len = self.max_seq_len,
            batch_size = self.batch_size,
            is_training = True)

        if training:
          current_time = datetime.now()
          self.estimator.train(
                          input_fn = self.train_input_fn,
                          max_steps = num_train_steps)
          print("Training took time ", datetime.now() - current_time)

        return


    def evaluate(self, eval_features, hyperparams, save_load_options, evaluating=True):

      self.train(features, training=False)

      self.test_input_fn = self.input_fn_builder(
            features = eval_features,
            max_seq_len = self.max_seq_len,
            batch_size = self.batch_size,
            is_training = False)

      if evaluating:
          current_time = datetime.now()
          test_result = self.estimator.evaluate(
                 input_fn = self.test_input_fn,
                 checkpoint_path = self.model_dir,
                 steps=None)

          print("Test took time ", datetime.now() - current_time)
          print(test_result)

def main():
    tr = TrainGPT(lr = 2.5e-4,
                  max_seq_len = 256,
                  batch_size = 16,
                  num_train_epochs = 1000,
                  num_warmup_steps = 10,
                  model_dir="./gpt2/test_model/")
    train_features = tr.load_features(0, 2000) # 2000 held-out log(perplexity)
    eval_features = tr.load_features(2000, 2500) # 500 held-out log(perplexity)
    tr.train(train_features, training=True)
    tr.evaluate(eval_features, evaluating=True)


if __name__ == '__main__':
    main()
