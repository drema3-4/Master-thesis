from openai import OpenAI

from master_thesis.llm.models import (
    LLMClientConfigSchema,
    GenerationConfigSchema
)


class LocalLLM:
    def __init__(
        self, config: LLMClientConfigSchema
    ) -> None:
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key
        )
        self.model = config.model

    def chat(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfigSchema
    ) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            seed=config.seed,
            extra_body={
                "top_k": config.top_k,
                "chat_template_kwargs": {
                    "enable_thinking": config.enable_thinking
                }
            }
        )

        message = response.choices[0].message

        return {
            "content": message.content,
            "reasoning_content": getattr(
                message,
                "reasoning_content",
                None
            ),
            "usage": (
                response.usage.model_dump()
                if response.usage
                else None
            ),
            "raw": response.model_dump(),
        }