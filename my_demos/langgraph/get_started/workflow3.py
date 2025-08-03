import os
import random
from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import Annotated

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver

# 需要运行目录有 .env 文件, 文件中有 OPENAI_API_KEY 和 OPENAI_API_BASE
load_dotenv()


# 如果使用 @tool 必须有文档注释, 也就是函数下方的 """xxx"""
@tool
def get_weather(location: str) -> str:
    """
    get weather from location
    :param location: a place
    :return: weather of location
    """
    available_weather = ['Sunny', 'Rain', 'Windy']
    return f'{location} is always {random.choice(available_weather)}'


@tool
def add(a, b) -> str:
    """
    add two numbers
    :param a: number 1
    :param b: number 2
    :return: str(a + b)
    """
    return f'{a + b}'


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class ChatBotWithToolGraph:
    def __init__(self) -> None:
        self.llm = init_chat_model(
            model="openai:gpt-4o-mini",
        )
        tools = [get_weather, add]
        self.llm = self.llm.bind_tools(tools)

        self.memory = InMemorySaver()

        graph_builder = StateGraph(State)
        graph_builder.add_node("chatbot", self._chat_bot_node)
        tool_node = ToolNode(tools)
        graph_builder.add_node("tools", tool_node)
        graph_builder.add_conditional_edges(
            "chatbot",
            tools_condition,
        )
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("tools", "chatbot")
        graph_builder.add_edge("chatbot", END)
        self.graph = graph_builder.compile(checkpointer=self.memory)

    def _chat_bot_node(self, state) -> dict:
        response = self.llm.invoke(state['messages'])
        return {"messages": [response]}

    def visualize(self, filename="graph.png"):
        try:
            img_data = self.graph.get_graph().draw_mermaid_png()
            with open(filename, "wb") as f:
                f.write(img_data)
            print(f"Graph image saved to: {filename}")
        except Exception as e:
            print("Visualization error:", e)

    def chat(self):
        config = {"configurable": {"thread_id": "1"}}
        while True:
            try:
                user_input = input("User: ")
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break

                response = self.graph.invoke(
                    {"messages": [{"role": "user", "content": user_input}]},
                    config,
                    stream_mode="values"
                )
                # print(response)
                print(f'Assistant: {response["messages"][-1].content}')

            except Exception as e:
                print(f"Error: {e}")
                break


if __name__ == '__main__':
    chatbot = ChatBotWithToolGraph()
    print('=' * 90)
    print('Start chatting... input `exit` / `quit` / `q` to quit.')
    print('=' * 90)
    chatbot.chat()
