from parlai.agents.contrastive_learning.common_things import (
    SampleExtendDictionaryAgent,
    observe_samp_expanded_observation,
    cl_batchify,
    _set_samp_label_vec,
    _set_samp_multi_turn_text_vec,
    cl_init,
    cl_share,
    cl_state_dict,
    cl_train,
    cl_eval_step,
    sample_batchify_multi_turn
)
from parlai.agents.dialog_wae.dialog_wae import DialogWaeAgent
from parlai.core.torch_generator_agent import Batch


class SampleExtendPersonDictionaryAgent(SampleExtendDictionaryAgent):
    def __init__(self, opt, shared=None):
        """Initialize DictionaryAgent."""
        super().__init__(opt, shared)
        if not shared:
            delimiter = opt.get('delimiter', '\n')
            self.add_token(delimiter)
            self.freq[delimiter] = 999999999

            if DialogWaeAgent.P1_TOKEN:
                self.add_token(DialogWaeAgent.P1_TOKEN)

            if DialogWaeAgent.P2_TOKEN:
                self.add_token(DialogWaeAgent.P2_TOKEN)

            if DialogWaeAgent.P1_TOKEN:
                self.freq[DialogWaeAgent.P1_TOKEN] = 999999998

            if DialogWaeAgent.P2_TOKEN:
                self.freq[DialogWaeAgent.P2_TOKEN] = 999999997


class OrigDialogWaeAgent(DialogWaeAgent):
    @staticmethod
    def dictionary_class():
        """
        Return the dictionary class that this agent expects to use.

        Can be overridden if a more complex dictionary is required.
        """
        return SampleExtendPersonDictionaryAgent

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id = 'CLDialogWaeAgent'
        self.sample_batchify_func = sample_batchify_multi_turn

    def observe(self, observation):
        """
        Process incoming message in preparation for producing a response.

        This includes remembering the past history of the conversation.
        """
        observation = observe_samp_expanded_observation(observation, multi_turn=True)
        return super().observe(observation)

    def compute_loss(self, batch, return_output=False, return_output_only=False):
        """
        Compute and return the loss for the given batch.

        Easily overridable for customized loss functions.

        If return_output is True, the full output from the call to self.model()
        is also returned, via a (loss, model_output) pair.
        """
        if batch.label_vec is None:
            raise ValueError('Cannot compute loss without a label.')
        if return_output_only:
            return self.model(*self._model_input(batch), ys=batch.label_vec,
                              res_lens=batch.label_lengths), batch.label_vec[:, 1:]
        else:
            return super().compute_loss(batch, return_output)


class CLHredAgent(OrigDialogWaeAgent):
    def __init__(self, opt, shared=None):
        self.use_external_ref_model = False
        super().__init__(opt, shared)
        if not opt.get('hred', False):
            raise RuntimeError('CL training is not applicable to Non-MLE methods.')
        if self.opt['naive_neg_sampling']:
            raise RuntimeError("Naive negtive sampling is not supported yet by CLHred!")
        cl_init(self, shared)

    def share(self):
        shared = super().share()
        return cl_share(self, shared)

    def state_dict(self):
        states = super().state_dict()
        return cl_state_dict(self, states)

    def vectorize(
        self,
        obs,
        history,
        add_start=True,
        add_end=True,
        text_truncate=None,
        label_truncate=None,
    ):
        _set_samp_multi_turn_text_vec(self, obs, text_truncate)
        _set_samp_label_vec(self, obs, add_start=True, add_end=True,
                            truncate=label_truncate)
        return super().vectorize(obs, history, add_start=add_start,
                                 add_end=add_end, text_truncate=text_truncate,
                                 label_truncate=label_truncate)

    def batchify(self, obs_batch, sort=False):
        batch: Batch = super().batchify(obs_batch, sort=sort)
        return cl_batchify(self, batch)

    def train_step(self, batch):
        cl_train(super(), self, batch)

    def eval_step(self, batch):
        return cl_eval_step(super(), self, batch)
