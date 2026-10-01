import os
import requests
import json

from dotenv import load_dotenv
from openai import OpenAI

# Load key
load_dotenv()

or_apiKey = os.getenv("OPENROUTER_API_KEY")

# Check key credit availability
url = "https://openrouter.ai/api/v1/credits"
headers = {
    "Authorization": or_apiKey
}

response = requests.get(url, headers=headers)
print(response.json())

# Smoke Test
client = OpenAI(
    base_url = "https://openrouter.ai/api/v1",
    api_key = os.getenv("OPENROUTER_API_KEY"),
)

## GPT-5
gpt5_response = client.chat.completions.create(
  extra_headers={},
  extra_body={},
  model="openai/gpt-5",
  stream=True,
  messages=[
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What is in this image?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "https://live.staticflickr.com/3851/14825276609_098cac593d_b.jpg"
          }
        }
      ]
    }
  ]
)
u = gpt5_response.model_dump()["usage"]
gpt5_cost = u.get("cost")

print("\nUser: Hi GPT-5. Can you see what is in this image? (Dolphins swimming)")
print(f"GPT-5: {gpt5_response.choices[0].message.content}")
print(f"GPT-5 cost: {gpt5_cost}")

## GPT-5-mini
gpt5mini_response = client.chat.completions.create(
  extra_headers={},
  extra_body={},
  model="openai/gpt-5-mini",
  stream=True,
  messages=[
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What is in this image?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "https://live.staticflickr.com/3851/14825276609_098cac593d_b.jpg"
          }
        }
      ]
    }
  ]
)
u = gpt5mini_response.model_dump()["usage"]
gpt5mini_cost = u.get("cost")

print("\nUser: Hi GPT-5-mini. Can you see what is in this image? (Dolphins swimming)")
print(f"GPT-5-mini: {gpt5mini_response.choices[0].message.content}")
print(f"GPT-5-mini cost: {gpt5mini_cost}")

## Qwen3-Coder-480B-A35B
qwen3Coder_response = client.chat.completions.create(
  extra_headers={},
  extra_body={},
  model="qwen/qwen3-coder",
  stream=True,
  messages=[
    {
      "role": "user",
      "content": "What is 7 + 7?"
    }
  ]
)
u = qwen3Coder_response.model_dump()["usage"]
qwen3Coder_cost = u.get("cost")

print("\nUser: What is 7 + 7? (14)")
print(f"Qwen3-Coder: {qwen3Coder_response.choices[0].message.content}")
print(f"Qwen3-Coder cost: {qwen3Coder_cost}")

## JEV

# The model answers narrow, typed questions about the state. Your code owns the workflow.
response = requests.post(
  url="https://openrouter.ai/api/alpha/decisions",
  headers={
    "Authorization": or_apiKey,
    "Content-Type": "application/json",
    },
  data=json.dumps({
    "model": "typesafe/jev-1.13",
    "state": "Help! My payouts have been failing for 3 days.",
    "questions": {
      "is_urgent": {
        "type": "noul",
        "instructions": "Does this message convey urgency?",
        "criteria": {
          "true": "Explicitly time-sensitive",
          "false": "No urgency expressed"
        }
      },
      "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
          "billing": "Payments, invoicing, refunds",
          "technical": "Bugs, outages, integrations",
          "sales": "Pricing, upgrades, new accounts"
        }
      },
      "frustration": {
        "type": "score",
        "instructions": "How frustrated is the customer?",
        "criteria": ["Calm", "Frustrated", "Very angry"]
      }
    }
  })
)

answers = response.json()["answers"]
# noul is a probability from 0 (no) to 1 (yes); choice and score carry the full distribution.
print(answers["is_urgent"]["noul"])
print(answers["department"]["choice"], answers["department"]["probabilities"])
print(answers["frustration"]["score"])
print(f"JEV cost: {answers["usage"]["cost"]}")