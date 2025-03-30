from transformers import Qwen2ForCausalLM, AutoTokenizer, Qwen2TokenizerFast
from my_demos.llm_demos import ChatBot

QWEN_HYPER_PARAM = {
    "system": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    "roles": ("user", "assistant"),
}


class QwenChatBot(ChatBot):
    def __init__(self, model_path, device="cpu"):
        print('### `QwenChatBot` Class init')
        system = QWEN_HYPER_PARAM["system"]
        roles = QWEN_HYPER_PARAM["roles"]
        self.device = device
        super().__init__(model_path, roles, system)

    def load_model(self, **kwargs):
        model = Qwen2ForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto"
        ).to(self.device)
        return model

    def load_tokenizer(self, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        return tokenizer

    def chat(self, prompt):
        cur_prompt = {
            "role": "user",
            "content": prompt
        }
        self._chat_history.append(cur_prompt)

        text = self.tokenizer.apply_chat_template(
            self._chat_history,
            tokenize=False,
            add_generation_prompt=True
        )

        print('=====')
        print(text)

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        self._chat_history.append({
            "role": "assistant",
            "content": response
        })

        return response

    def clean_history(self, text):
        self._chat_history = [self._chat_history[0]]