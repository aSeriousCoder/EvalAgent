import numpy as np
from openai import OpenAI


class LLMFrame:
    def __init__(self, llm_name):
        self.llm_name = llm_name
        self.embedding_model = EmbeddingModel()
        if self.llm_name in OpenAIModel.model_list():
            self.llm_model = OpenAIModel(llm_name)
        elif self.llm_name in DeepSeekModel.model_list():
            self.llm_model = DeepSeekModel(llm_name)
        elif self.llm_name in QwenModel.model_list():
            self.llm_model = QwenModel(llm_name)
        else:
            raise ValueError(f"Invalid LLM name: {llm_name}")

    def generate(self, message):
        response = self.llm_model.generate(message)
        return response

    def embed(self, message):
        return self.embedding_model.embedding(message)


class OpenAIModel:
    def __init__(self, llm_name):
        self.llm_name = llm_name
        self.client = OpenAI(
            api_key="<YOUR_OPENAI_API_KEY>"
        )

    @staticmethod
    def model_list():
        return ["gpt-4o"]

    def generate(self, message):
        response = self.client.chat.completions.create(
            model=self.llm_name, messages=[{"role": "user", "content": message}]
        )
        return response.choices[0].message.content


class DeepSeekModel:
    def __init__(self, llm_name):
        self.llm_name = llm_name
        self.client = OpenAI(
            api_key="<YOUR_DEEPSEEK_API_KEY>",
            base_url="https://api.deepseek.com",
        )

    @staticmethod
    def model_list():
        return ["deepseek-chat"]

    def generate(self, message):
        response = self.client.chat.completions.create(
            model=self.llm_name, messages=[{"role": "user", "content": message}]
        )
        return response.choices[0].message.content


class QwenModel:
    def __init__(self, llm_name):
        self.llm_name = llm_name
        self.client = OpenAI(
            api_key="<YOUR_BAILIAN_API_KEY>",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    @staticmethod
    def model_list():
        return ["qwen3-235b-a22b"]

    def generate(self, message):
        response = self.client.chat.completions.create(
            model=self.llm_name,
            messages=[{"role": "user", "content": message}],
            extra_body={"enable_thinking": False},
        )
        return response.choices[0].message.content


class EmbeddingModel:
    def __init__(self):
        self.openai_client = OpenAI(
            api_key="<YOUR_OPENAI_API_KEY>"
        )

    def embedding(self, sentences):
        """
        @param sentences: list[str]
        @return: np.array of shape (1536,)
        """
        response = self.openai_client.embeddings.create(
            input=sentences, model="text-embedding-3-small"
        )
        embeddings = [data.embedding for data in response.data]
        return np.array(embeddings)
