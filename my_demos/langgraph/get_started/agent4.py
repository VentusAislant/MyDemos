from langchain_core.messages import ToolMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
import os
from dotenv import load_dotenv

# 需要运行目录有 .env 文件, 文件中有 OPENAI_API_KEY 和 OPENAI_API_BASE
load_dotenv()

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

checkpointer = InMemorySaver()

agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather],
    checkpointer=checkpointer
)

# Run the agent
config = {"configurable": {"thread_id": "1"}}

sf_response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
    config
)
ny_response = agent.invoke(
    {"messages": [{"role": "user", "content": "what about new york?"}]},
    config
)


state = checkpointer.get_tuple(config) # 传入 thread_id

# 然后查看存储的 message history（你可以打印整个 state 看它结构）
# print(state)

for msg in state.checkpoint['channel_values']["messages"]:
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

"""
==============================
Message type: HumanMessage
Content: what is the weather in sf
==============================
Message type: AIMessage
Tool Calls:
  Tool name: get_weather
  Args: {'city': 'San Francisco'}
==============================
Message type: ToolMessage
Content: It's always sunny in San Francisco!
Tool name: get_weather
==============================
Message type: AIMessage
Content: The weather in San Francisco is always sunny!
==============================
Message type: HumanMessage
Content: what about new york?
==============================
Message type: AIMessage
Tool Calls:
  Tool name: get_weather
  Args: {'city': 'New York'}
==============================
Message type: ToolMessage
Content: It's always sunny in New York!
Tool name: get_weather
==============================
Message type: AIMessage
Content: The weather in New York is also always sunny!
"""