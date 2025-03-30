import argparse
import os
import yaml

from my_demos.llm_demos.tiny_llama_demo import TinyLlamaChatBot
from my_demos.llm_demos.qwen_demo import QwenChatBot


def parser_config(conf_file):
    assert os.path.exists(conf_file) and (conf_file.endswith(".yaml") or conf_file.endswith(".yml"))

    with open(conf_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def main(args=None):
    # 解析配置文件
    config = parser_config(args.config_path)
    print(f"Configuration loaded from {args.config_path}:")
    print(config)

    # 你可以根据不同的参数来初始化不同的模型
    # 例如，如果 llm_type 是 qwen，则使用 QwenChatBot 初始化

    if args.llm_type == 'qwen':
        qwen_cfg = config['qwen']
        model_path = qwen_cfg[args.llm_name]
        chatbot = QwenChatBot(model_path, args.device)
    elif args.llm_type == 'tinyllama':
        tinyllama_cfg = config['tinyllama']
        model_path = tinyllama_cfg[args.llm_name]
        chatbot = TinyLlamaChatBot(model_path, args.device)
    else:
        raise NotImplementedError

    # 之后你可以执行你的对话逻辑等
    if args.mode == 'multi':
        print('### welcome to chatbot, input `quit` or `exit` to exit.')
        while True:
            prompt = input('User >>> ')
            if prompt == 'quit':
                break
            resp = chatbot.chat(prompt)
            print("Assistant >>>", resp)
    elif args.mode == 'single':
        print('### welcome to chatbot')
        prompt = input('User >>> ')
        resp = chatbot.chat(prompt)
        print("Assistant >>>", resp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ChatBot Configuration and Initialization")

    # 参数1：config_path，指向一个yaml配置文件
    parser.add_argument(
        '--config_path',
        type=str, default="configs/llm_demos_conf.yaml",
        help="Path to the configuration YAML file")

    # 参数2：llm类型，可以选择 qwen, llama, tinyllama
    parser.add_argument('--llm_type', type=str, choices=['qwen', 'llama', 'tinyllama'], required=True, help="LLM type")

    # 参数3：设备，可以选择 cuda 或 cpu
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cpu',
                        help="Device to use (cuda or cpu)")

    # 参数4：llm name
    parser.add_argument('--llm_name', type=str, required=True, help="Name of the LLM")

    # 参数5：mode：对话模式，可以是多轮或者单轮
    parser.add_argument('--mode', type=str, choices=['multi', 'single'], default='multi',
                        help="Conversation mode (multi or single)")

    # 解析命令行参数
    args = parser.parse_args()
    main(args)
