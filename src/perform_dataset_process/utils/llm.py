import uuid


def generate_id():
    return str(uuid.uuid4())


vlm_dict = {
    "custom_id": "{id}",
    "method": "POST",
    "url": "/v1/chat/completions",
    "body": {
        "model": "{model_name}",
        "messages": [
            {"role": "system", "content": "{system_prompt}"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,{base64_image}"},
                    },
                    {"type": "text", "text": "{text}"},
                ],
            },
        ],
    },
}


llm_dict = {
    "custom_id": "{id}",
    "method": "POST",
    "url": "/v1/chat/completions",
    "body": {
        "model": "{model_name}",
        "messages": [
            {"role": "system", "content": "{system_prompt}"},
            {"role": "user", "content": "{text}"},
        ],
    },
}


def format_vlm_message(
    system_prompt, base64_str, text, model_name: str = "qwen-vl-max", id=None
):
    if id is None:
        id = generate_id()
    return {
        "custom_id": id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_str}"},
                        },
                        {"type": "text", "text": text},
                    ],
                },
            ],
        },
    }


def format_llm_message(system_prompt, text, model_name: str = "deepseek-r1", id=None):
    if id is None:
        id = generate_id()
    return {
        "custom_id": id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
        },
    }
