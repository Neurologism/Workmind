import requests
from env import WEBHOOK_URL

# embed generator: https://message.style/app/editor


def webhook_training_error(task_id, error):
    webhook_post(
        {
            "id": 652627557,
            "title": "Error during training",
            "color": 15409955,
            "fields": [
                {
                    "id": 231865094,
                    "name": "Task ID",
                    "value": f"```\n{task_id}\n```",
                },
                {
                    "id": 367956011,
                    "name": "Error",
                    "value": f"```json\n{error}\n```",
                },
            ],
        }
    )


def webhook_post(embed):
    if not WEBHOOK_URL:
        return
    response = requests.post(
        WEBHOOK_URL,
        json={
            "content": "",
            "embeds": [embed],
            "components": [],
            "actions": {},
        },
    )
    if response.status_code != 204:
        print(f"Failed to post webhook: {response.status_code} {response.text}")
