# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
from ncel.utils.layers import the_gpu
import os
from ncel.utils.misc import recursively_set_device

def get_checkpoint_path(FLAGS, suffix=".ckpt", best=False):
    # Set checkpoint path.

    if FLAGS.eval_only_mode and FLAGS.eval_only_mode_use_best_checkpoint:
        best = True

    if FLAGS.ckpt_path.endswith(".ckpt") or FLAGS.ckpt_path.endswith(".ckpt_best"):
        checkpoint_path = FLAGS.ckpt_path
    else:
        checkpoint_path = os.path.join(FLAGS.ckpt_path, FLAGS.experiment_name + suffix)
    if best:
        checkpoint_path += "_best"
    return checkpoint_path


class ModelTrainer(object):
    def __init__(self, model, logger, epoch_length, vocabulary, FLAGS):
        self.model = model
        self.logger = logger
        self.epoch_length = epoch_length
        self.word_vocab, self.entity_vocab, self.id2wiki_vocab = vocabulary

        self.logger.Log('One epoch is ' + str(self.epoch_length) + ' steps.')

        self.dense_parameters = [param for name, param in model.named_parameters() if name not in ["embed.embed.weight"]]
        self.sparse_parameters = [param for name, param in model.named_parameters() if name in ["embed.embed.weight"]]
        self.optimizer_type = FLAGS.optimizer_type
        self.l2_lambda = FLAGS.l2_lambda
        self.ckpt_step = FLAGS.ckpt_step
        self.ckpt_on_best_dev_error = FLAGS.ckpt_on_best_dev_error
        self.learning_rate_decay_when_no_progress = FLAGS.learning_rate_decay_when_no_progress
        self.training_data_length = None
        self.eval_interval_steps = FLAGS.eval_interval_steps

        self.step = 0
        self.best_dev_error = 1.0
        self.best_dev_step = 0

        # record best dev, test acc
        self.best_dev_cacc = 0
        self.best_dev_macc = 0
        self.best_dev_dacc = 0
        self.best_test_cacc = 0
        self.best_test_macc = 0
        self.best_test_dacc = 0

        # GPU support.
        self.gpu = FLAGS.gpu
        the_gpu.gpu = FLAGS.gpu
        if self.gpu >= 0:
            model.cuda()
        else:
            model.cpu()

        self.optimizer_reset(FLAGS.learning_rate)

        self.standard_checkpoint_path = get_checkpoint_path(FLAGS)
        self.best_checkpoint_path = get_checkpoint_path(FLAGS, best=True)

        # Load checkpoint if available.
        if FLAGS.load_best and os.path.isfile(self.best_checkpoint_path):
            self.logger.Log("Found best checkpoint, restoring.")
            self.load(self.best_checkpoint_path, cpu=FLAGS.gpu < 0)
            self.logger.Log(
                "Resuming at step: {} with best dev accuracy: {}".format(
                    self.step, 1. - self.best_dev_error))
        elif os.path.isfile(self.standard_checkpoint_path):
            self.logger.Log("Found checkpoint, restoring.")
            self.load(self.standard_checkpoint_path, cpu=FLAGS.gpu < 0)
            self.logger.Log(
                "Resuming at step: {} with best dev accuracy: {}".format(
                    self.step, 1. - self.best_dev_error))


    def optimizer_reset(self, learning_rate):
        self.learning_rate = learning_rate

        if self.optimizer_type == "Adam":
            self.optimizer = optim.Adam(self.dense_parameters, lr=learning_rate,
                weight_decay=self.l2_lambda)

            if len(self.sparse_parameters) > 0:
                self.sparse_optimizer = optim.SparseAdam(self.sparse_parameters, lr=learning_rate)
            else:
                self.sparse_optimizer = None
        elif self.optimizer_type == "SGD":
            self.optimizer = optim.SGD(self.dense_parameters, lr=learning_rate,
                weight_decay=self.l2_lambda)
            if len(self.sparse_parameters) > 0:
                self.sparse_optimizer = optim.SGD(self.sparse_parameters, lr=learning_rate)
            else:
                self.sparse_optimizer = None

    def optimizer_step(self):
        self.optimizer.step()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.step()
        self.step += 1

    def optimizer_zero_grad(self):
        self.optimizer.zero_grad()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.zero_grad()

    def new_accuracy(self, dev_acc, test_acc=None):
        # Track best dev error
        dev_cacc, dev_macc, dev_dacc = dev_acc
        if test_acc is not None:
            test_cacc, test_macc, test_dacc = test_acc
        if (1 - dev_macc) < 0.99 * self.best_dev_error:
            self.best_dev_error = 1 - dev_macc
            self.best_dev_step = self.step
            if self.ckpt_on_best_dev_error and self.step > self.ckpt_step:
                self.logger.Log(
                    "Checkpointing with new best dev accuracy of %f" %
                    dev_macc)
                self.save(self.best_checkpoint_path)
            self.best_dev_cacc = dev_cacc
            self.best_dev_macc = dev_macc
            self.best_dev_dacc = dev_dacc
            if test_acc is not None:
                self.best_test_cacc = test_cacc
                self.best_test_macc = test_macc
                self.best_test_dacc = test_dacc

        # Learning rate decay
        if self.learning_rate_decay_when_no_progress != 1.0:
            last_epoch_start = self.step - (self.step % self.epoch_length)
            if self.step - last_epoch_start <= self.eval_interval_steps and self.best_dev_step < (last_epoch_start - self.epoch_length):
                    self.logger.Log('No improvement after one epoch. Lowering learning rate.')
                    self.optimizer_reset(self.learning_rate * self.learning_rate_decay_when_no_progress)

    def checkpoint(self):
        self.logger.Log("Checkpointing.")
        self.save(self.standard_checkpoint_path)

    def save(self, filename):
        if the_gpu() >= 0:
            recursively_set_device(self.model.state_dict(), gpu=-1)
            recursively_set_device(self.optimizer.state_dict(), gpu=-1)

        # Always sends Tensors to CPU.
        save_dict = {
            'step': self.step,
            'best_dev_error': self.best_dev_error,
            'best_dev_step': self.best_dev_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'word_vocab': self.word_vocab,
            'entity_vocab': self.entity_vocab,
            'id2wiki_vocab': self.id2wiki_vocab
            }
        if self.sparse_optimizer is not None:
            save_dict['sparse_optimizer_state_dict'] = self.sparse_optimizer.state_dict()
        torch.save(save_dict, filename)

        if the_gpu() >= 0:
            recursively_set_device(self.model.state_dict(), gpu=the_gpu())
            recursively_set_device(self.optimizer.state_dict(), gpu=the_gpu())

    def load(self, filename, cpu=False):
        if cpu:
            # Load GPU-based checkpoints on CPU
            checkpoint = torch.load(
                filename, map_location=lambda storage, loc: storage)
        else:
            checkpoint = torch.load(filename)
        model_state_dict = checkpoint['model_state_dict']

        if 'embed.embed.weight' in model_state_dict:
            loaded_embeddings = model_state_dict['embed.embed.weight']
            del(model_state_dict['embed.embed.weight'])

            count = 0
            for ent_id in checkpoint['entity_vocab']:
                if ent_id in self.entity_vocab:
                    self_index = self.entity_vocab[ent_id]
                    loaded_index = checkpoint['entity_vocab'][ent_id]
                    self.model.embed.embed.weight.data[self_index, :] = loaded_embeddings[loaded_index, :]
                    count += 1
            self.logger.Log('Restored ' + str(count) + ' entity from checkpoint.')

        self.model.load_state_dict(model_state_dict, strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.load_state_dict(checkpoint['sparse_optimizer_state_dict'])

        self.step = checkpoint['step']
        self.best_dev_step = checkpoint['best_dev_step']
        self.best_dev_error = checkpoint['best_dev_error']
