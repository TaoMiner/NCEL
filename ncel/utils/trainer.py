class ModelTrainer(object):
    def __init__(self, model, logger, epoch_length, vocabulary, FLAGS):
        self.model = model