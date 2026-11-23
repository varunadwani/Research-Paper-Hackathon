import os
from openai import OpenAI

client = OpenAI(
    base_url="http://10.7.25.115:30400/v1",
    api_key=os.environ["HACKATHON_API_KEY"],
)

response = client.chat.completions.create(
    model="qwen3-30b-a3b",
    messages=[{"role": "user", "content": "Reply with: API connection works"}],
    max_tokens=1200,
)

message = response.choices[0].message
print("Finish reason:", response.choices[0].finish_reason)
print("Content:", repr(message.content))
print("Reasoning:", repr(getattr(message, "reasoning_content", None)))