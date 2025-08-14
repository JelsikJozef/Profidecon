import os
import json

class LLMPicker:
    def __init__(self, backend: str = "openai"):
        self.backend = backend
        if backend == "ollama":
            self.llm_model = os.getenv("OLLAMA_MODEL", "gpt-oss-20b")
        elif backend == "huggingface":
            self.llm_model = os.getenv("HF_MODEL", "Milos/slovak-gpt-j-405M")
        else:  # openai
            self.llm_model = "gpt-4o-mini"

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

        elif self.backend == "huggingface":
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                import torch

                # Load the Slovak GPT-J model
                tokenizer = AutoTokenizer.from_pretrained(self.llm_model)
                model = AutoModelForCausalLM.from_pretrained(self.llm_model)

                # Set padding token if not available
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                # Create input text with Slovak prompt
                input_text = f"Úloha: Zhrň nasledujúci text v slovenčine a vytvor 5 krátkych tagov.\n\nText: {snippet}\n\nOdpoveď (JSON formát s kľúčmi 'summary' a 'tags'):"

                # Tokenize input
                inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)

                # Generate response
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        max_new_tokens=300,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        attention_mask=inputs.attention_mask
                    )

                # Decode response
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Extract only the generated part (after the input prompt)
                response = generated_text[len(input_text):].strip()

                # Try to extract JSON from response
                try:
                    # Look for JSON-like structure
                    start_idx = response.find('{')
                    end_idx = response.rfind('}') + 1

                    if start_idx != -1 and end_idx != 0:
                        json_str = response[start_idx:end_idx]
                        data = json.loads(json_str)
                        summary = data.get("summary", "")
                        tags = data.get("tags", [])

                        if isinstance(tags, str):
                            tags = [t.strip() for t in tags.split(",") if t.strip()]

                        return summary, tags
                    else:
                        # Fallback: try to parse the response as simple text
                        lines = response.split('\n')
                        summary = ""
                        tags = []

                        for line in lines:
                            line = line.strip()
                            if line and not summary:
                                summary = line
                            elif line and len(tags) < 5:
                                # Try to extract tags from line
                                if any(word in line.lower() for word in ['tag', 'kľúč', 'kategória']):
                                    continue
                                tags.append(line.replace('-', '').replace('*', '').strip())

                        return summary[:200], tags[:5]  # Limit length

                except json.JSONDecodeError:
                    # Simple fallback - use first part as summary
                    words = response.split()
                    if len(words) > 10:
                        summary = ' '.join(words[:20])
                        # Generate simple tags from the text
                        tags = [word.lower() for word in words[20:25] if len(word) > 3]
                        return summary, tags[:5]

                return "", []

            except Exception as e:
                print(f"Error generating summary and tags (HuggingFace): {e}")
                return "", []
        else:
            print(f"Unknown LLM backend: {self.backend}")
            return "", []
