from openai import OpenAI

from master_thesis.schemas.llm_schemas import (
    LLMConfigShema
)


class LocalLLM:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str
    ) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def chat(
        self,
        messages: list[dict[str, str]],
        cfg: LLMConfigShema
    ) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            max_tokens=cfg["max_tokens"],
            seed=cfg["seed"],
            extra_body={
                "top_k": cfg["top_k"],
                "chat_template_kwargs": {
                    "enable_thinking": cfg["enable_thinking"]
                }
            }
        )

        msg = response.choices[0].message

        return {
            "content": msg.content,
            "reasoning_content": getattr(msg, "reasoning_content", None),
            "usage": response.usage.model_dump() if response.usage else None,
            "raw": response.model_dump(),
        }