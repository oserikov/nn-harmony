import dynet as dy
import numpy as np


class ModelDyNetBuilder:
    _input_l = dy.Expression

    LAYERS = 1

    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.001):
        self._model = dy.ParameterCollection()

        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim

        self._rnn = dy.SimpleRNNBuilder(self.LAYERS, self._input_dim, self._hidden_dim, self._model)
        # self._rnn.disable_dropout()
        self._W = self._model.add_parameters((self._output_dim, self._hidden_dim), init=dy.NormalInitializer())

        self._learning_rate = learning_rate
        self._trainer = dy.MomentumSGDTrainer(self._model, learning_rate=self._learning_rate)

        self._l2_param = 0.0006
        # self._l2_param = 0.0

        self._init_layers()

    def _batch_forward(self, training_vectors):
        batch_loss = []
        predictions = []
        for input_vector, target_vector in training_vectors:
            pred = dy.softmax(self._forward(input_vector))

            loss = -dy.log(dy.pick(pred, target_vector.index(1))) \
                   + self._l2_param * sum([dy.squared_norm(param) for param in self._model.parameters_list()])
            batch_loss.append(loss)
            predictions.append(pred.npvalue())

        return predictions, dy.esum(batch_loss), np.mean([loss.npvalue() for loss in batch_loss])

    def _forward(self, input_vector):
        self._input_l = self._input_l.add_input(dy.inputVector(input_vector))

        hidden_layer_output = self._input_l.output()
        self._context_state_vector = hidden_layer_output.npvalue()

        output_values = self._W * hidden_layer_output

        return output_values

    def train_batches(self, training_batches):
        return self._process_batches(training_batches, learning=True)

    def predict_batches(self, training_batches):
        return self._process_batches(training_batches, learning=False)

    def _process_batches(self, training_batches, learning: bool):
        epoch_loss = []
        predictions = []
        for training_batch in training_batches:
            self._reload_model_into_cg()

            batch_predictions, loss, mean_loss = self._batch_forward(training_batch)

            if learning:
                self._process_loss(loss)
                self._trainer.update()

            epoch_loss.append(mean_loss)
            predictions.append(batch_predictions)

        return predictions, np.mean(epoch_loss)

    @staticmethod
    def _process_loss(loss):
        loss.backward()

    def _reload_model_into_cg(self):
        dy.renew_cg()
        self._init_layers()
        # pass

    def _init_layers(self):
        self._input_l = self._rnn.initial_state()

    def get_context_state(self):
        return self._context_state_vector
