import os
from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import Annotated

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

# 需要运行目录有 .env 文件, 文件中有 OPENAI_API_KEY 和 OPENAI_API_BASE
load_dotenv()


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class ChatBotGraph:
    def __init__(self) -> None:
        self.llm = init_chat_model(
            model="openai:gpt-4o-mini",
        )

        self.state = None

        graph_builder = StateGraph(State)
        graph_builder.add_node("chatbot", self._chat_bot_node)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)
        self.graph = graph_builder.compile()

    def _init_state(self, system_prompt=None):
        if system_prompt is None:
            system_prompt = "Never answer about weather!"
        self.state = State(
            messages=[SystemMessage(content=system_prompt)],
        )

    def _chat_bot_node(self, state) -> dict:
        # print('-' * 90)
        # print(state["messages"])
        response = self.llm.invoke(state['messages'])
        return {"messages":[response]}

    def visualize(self, filename="graph.png"):
        try:
            img_data = self.graph.get_graph().draw_mermaid_png()
            with open(filename, "wb") as f:
                f.write(img_data)
            print(f"Graph image saved to: {filename}")
        except Exception as e:
            print("Visualization error:", e)

    def chat(self, system_prompt=None):
        self._init_state(system_prompt)
        while True:
            try:
                user_input = input("User: ")
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break
                self.state['messages'].append(
                    HumanMessage(content=user_input)
                )

                # 必须将结果赋值给 self.state 来更新 self.state
                self.state = self.graph.invoke(self.state)
                print('Assistant: ', self.state['messages'][-1].content)
            except Exception as e:
                print(f"Error: {e}")
                break


if __name__ == '__main__':
    chatbot = ChatBotGraph()
    # print('Visualizing...')
    # chatbot.visualize()
    print('=' * 90)
    print('Start chatting... input `exit` / `quit` / `q` to quit.')
    print('=' * 90)
    chatbot.chat()
