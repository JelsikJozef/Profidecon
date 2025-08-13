import os
import json

class LLMPicker:
    def __init__(self, backend: str = "openai"):
        self.backend = backend
        self.llm_model = os.getenv("OLLAMA_MODEL", "gpt-oss-20b") if backend == "ollama" else "gpt-4o-mini"

    def generate_summary_and_tags(self, text: str) -> tuple[str, list[str]]:
        snippet = text[:1000]
        prompt = (
            "Summarize the provided text in Slovak and extract up to 5 short tags. "
            "Respond in Slovak with JSON using keys 'summary' and 'tags'."
        )
        if self.backend == "openai":
            try:
                from openai import OpenAI
                if not os.getenv("OPENAI_API_KEY"):
                    raise RuntimeError("OpenAI API key not found.")
                client = OpenAI()
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": snippet},
                ]
                resp = client.chat.completions.create(
                    model=self.llm_model,
                    messages=messages,
                    max_tokens=500,
                    temperature=0.0,
                )
                if not resp or not resp.choices:
                    raise ValueError("No response from LLM.")
                content = resp.choices[0].message.content.strip()
                content = content.replace("```json", "").replace("```", "").strip()
                data = json.loads(content)
                summary = data.get("summary", "")
                tags = data.get("tags", [])
                if isinstance(tags, str):
                    tags = [t.strip() for t in tags.split(",") if t.strip()]
                return summary, tags
            except Exception:
                print("Error generating summary and tags (OpenAI)")
                return "", []
        elif self.backend == "ollama":
            try:
                import requests
                ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
                payload = {
                    "model": self.llm_model,
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": snippet},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.0}
                }
                resp = requests.post(f"{ollama_url}/api/chat", json=payload, timeout=60)
                resp.raise_for_status()
                result = resp.json()
                content = result.get("message", {}).get("content", "").strip()
                content = content.replace("```json", "").replace("```", "").strip()
                data = json.loads(content)
                summary = data.get("summary", "")
                tags = data.get("tags", [])
                if isinstance(tags, str):
                    tags = [t.strip() for t in tags.split(",") if t.strip()]
                return summary, tags
            except Exception:
                print("Error generating summary and tags (Ollama)")
                return "", []
        else:
            print(f"Unknown LLM backend: {self.backend}")
            return "", []

