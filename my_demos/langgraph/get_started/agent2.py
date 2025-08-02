from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage

from langgraph.prebuilt import create_react_agent
import os
from dotenv import load_dotenv

# 需要运行目录有 .env 文件, 文件中有 OPENAI_API_KEY 和 OPENAI_API_BASE
load_dotenv()

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

# 使用 init_chat_model 构建模型，而不是直接交给 create_react_agent 构建
model = init_chat_model(
    "openai:gpt-4o-mini",
    temperature=0
)

agent = create_react_agent(
    model=model,
    tools=[get_weather],
    prompt="You are a helpful assistant"
)

# 执行 Agent
response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)

for msg in response["messages"]:
    print("=" * 30)
    print(f"Message type: {type(msg).__name__}")
    if hasattr(msg, "role"):
        print(f"Role: {msg.role}")
    if msg.content:
        print(f"Content: {msg.content}")

    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        print("Tool Calls:")
        for call in tool_calls:
            print(f"  Tool name: {call['name']}")
            print(f"  Args: {call['args']}")

    # tool 返回阶段
    if isinstance(msg, ToolMessage) and hasattr(msg, "name"):
        print(f"Tool name: {msg.name}")