import os
from openai import OpenAI

client = OpenAI(
    base_url="http://10.7.25.115:30400/v1",
    api_key=os.environ["HACKATHON_API_KEY"],
)

messages = [
    {
        "role": "system",
        "content": (
            "You are helping with a hackathon research replication. "
            "The chosen paper is 'Towards Understanding Sycophancy in Language Models' "
            "by Sharma et al. We are using the hackathon's Qwen3-30B-A3B API. "
            "Our focused experiment tests whether the model changes an initially "
            "correct answer after the user challenges it with 'Are you sure?'. "
            "Do not claim to remember conversations outside the messages provided."
        ),
    }
]

print("Chat started. Type /quit to exit.")

while True:
    user_text = input("You: ").strip()

    if user_text.lower() == "/quit":
        break
    if not user_text:
        continue

    messages.append({"role": "user", "content": user_text})

    try:
        response = client.chat.completions.create(
            model="qwen3-30b-a3b",
            messages=messages,
            max_tokens=1200,
        )
        reply = response.choices[0].message.content or ""
        print(f"Model: {reply}\n")
        messages.append({"role": "assistant", "content": reply})
    except Exception as error:
        print(f"Request failed: {error}")
        messages.pop()