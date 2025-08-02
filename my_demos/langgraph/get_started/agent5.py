from langchain_core.messages import ToolMessage
from langgraph.prebuilt import create_react_agent

import os
from pydantic import BaseModel
from dotenv import load_dotenv

# 需要运行目录有 .env 文件, 文件中有 OPENAI_API_KEY 和 OPENAI_API_BASE
load_dotenv()


class WeatherResponse(BaseModel):
    city: str
    weather: str

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather],
    response_format=WeatherResponse
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in new york?"}]}
)

structured_response = response['structured_response']
print(type(structured_response))
print(structured_response)
"""
<class '__main__.WeatherResponse'>
city='New York' weather='sunny'
"""