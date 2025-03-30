import abc
from my_demos.common import print_title, print_title_end


class ChatBot(abc.ABC):
    def __init__(self, model_path, roles, system, **kwargs):
        print('### `ChatBot` Class init')
        self._model_path = model_path
        self._roles = roles
        self._system = system
        self._chat_history = [{"role": "system", "content": self._system}]

        print_title("ChatBot Infos")
        print("model_path:", self._model_path)
        print("roles:", self._roles)
        print("system:", self._system)
        print("chat_history:", self._chat_history)

        print_title(f"Loading Model")
        if self._model_path is None:
            raise ValueError("`model_path` is required, but None given")
        self.model = self.load_model()
        print(f'model type: {self.model.__class__.__name__}')
        print(self.model)

        print_title(f"Loading Tokenizer")
        self.tokenizer = self.load_tokenizer()
        print(f'tokenizer type: {self.tokenizer.__class__.__name__}')
        print(f'vocab_size: {self.tokenizer.vocab_size}')
        print(f'model_max_length: {self.tokenizer.model_max_length}')
        print(f'padding_side: {self.tokenizer.padding_side}')
        print(f'truncation_side: {self.tokenizer.truncation_side}')
        print(f'eos_token | eos_token id : {self.tokenizer.eos_token} | {self.tokenizer.eos_token_id}')
        print(f'pad_token | pad_token id : {self.tokenizer.pad_token} | {self.tokenizer.pad_token_id}')
        print_title_end()

    @property
    def model_path(self):
        return self._model_path

    @property
    def system(self):
        return self._system

    @property
    def roles(self):
        return self._roles

    @property
    def chat_history(self):
        return self._chat_history

    @abc.abstractmethod
    def load_model(self, **kwargs):
        pass

    @abc.abstractmethod
    def load_tokenizer(self, **kwargs):
        pass

    @abc.abstractmethod
    def chat(self, prompt):
        pass

    @abc.abstractmethod
    def clean_history(self, text):
        pass
