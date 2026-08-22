from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "tam-v3-100m-instruct-chat"
VOLUME_NAME = "tam-research-data"
CHECKPOINT = "/vol/full100m-runs/TAM-v3-100M-2B-seed8100/final_dpo.pt"
SEED = 8100

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.7,<2.11",
        "transformers>=4.55,<5",
        "tokenizers>=0.21,<1",
        "numpy>=2.0,<3",
        "fastapi[standard]>=0.115,<1",
    )
    .add_local_python_source("tam_research")
)


def _normalize_history(history: list[dict[str, str]] | None) -> list[dict[str, str]]:
    clean: list[dict[str, str]] = []
    for item in history or []:
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            clean.append({"role": role, "content": content})
    return clean[-12:]


def _format_prompt(message: str, history: list[dict[str, str]] | None = None) -> str:
    parts: list[str] = []
    for item in _normalize_history(history):
        role = "User" if item["role"] == "user" else "Assistant"
        parts.append(f"{role}:\n{item['content']}\n")
    parts.append(f"User:\n{message.strip()}\nAssistant:\n")
    return "".join(parts)


@app.cls(
    image=image,
    gpu="L4",
    cpu=4,
    memory=16384,
    timeout=900,
    volumes={"/vol": volume},
)
class TAM100MInstruct:
    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import AutoTokenizer
        from tam_research.posttrain import load_checkpoint_model

        volume.reload()
        checkpoint = Path(CHECKPOINT)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"final instruct checkpoint not found: {checkpoint}; "
                "refusing to fall back to base or SFT"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("TAM 100M interactive inference requires CUDA")

        self.torch = torch
        self.device = torch.device("cuda")
        self.model = load_checkpoint_model(str(checkpoint), self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
        self.eos_id = self.tokenizer.eos_token_id
        self.max_seq_len = int(self.model.cfg.max_seq_len)

    def _generate(
        self,
        message: str,
        history: list[dict[str, str]] | None,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        seed: int,
    ) -> dict:
        torch = self.torch
        text = message.strip()
        if not text:
            raise ValueError("message must not be empty")

        max_new_tokens = max(1, min(int(max_new_tokens), 256))
        temperature = float(temperature)
        top_k = max(0, min(int(top_k), 200))
        prompt = _format_prompt(text, history)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        prompt_room = max(1, self.max_seq_len - max_new_tokens)
        prompt_ids = prompt_ids[-prompt_room:]
        tokens = torch.tensor(prompt_ids, device=self.device, dtype=torch.long).unsqueeze(0)
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        generated: list[int] = []

        with torch.inference_mode():
            for _ in range(max_new_tokens):
                context = tokens[:, -self.max_seq_len :]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = self.model(context)[:, -1, :].float()

                if temperature <= 0:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    logits = logits / max(temperature, 1e-4)
                    if top_k > 0:
                        values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
                        probs = torch.softmax(values, dim=-1)
                        sampled = torch.multinomial(probs, 1, generator=generator)
                        next_token = indices.gather(-1, sampled)
                    else:
                        probs = torch.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, 1, generator=generator)

                token_id = int(next_token.item())
                if token_id == self.eos_id:
                    break
                generated.append(token_id)
                tokens = torch.cat((tokens, next_token), dim=1)

        response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return {
            "response": response,
            "checkpoint": CHECKPOINT,
            "stage": "dpo-instruct",
            "max_seq_len": self.max_seq_len,
            "generated_tokens": len(generated),
        }

    @modal.method()
    def generate(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 40,
        seed: int = SEED,
    ) -> dict:
        return self._generate(message, history, max_new_tokens, temperature, top_k, seed)

    @modal.fastapi_endpoint(
        method="POST",
        docs=True,
        requires_proxy_auth=True,
        label="tam-v3-100m-instruct",
    )
    def chat(self, item: dict) -> dict:
        return self._generate(
            str(item.get("message", "")),
            item.get("history"),
            int(item.get("max_new_tokens", 128)),
            float(item.get("temperature", 0.7)),
            int(item.get("top_k", 40)),
            int(item.get("seed", SEED)),
        )


@app.local_entrypoint()
def main(
    prompt: str = "",
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_k: int = 40,
):
    model = TAM100MInstruct()
    if prompt.strip():
        result = model.generate.remote(
            message=prompt,
            history=[],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=SEED,
        )
        print(result["response"])
        return

    history: list[dict[str, str]] = []
    print("TAM-v3-100M-2B DPO instruct chat. Type /exit to quit.")
    while True:
        try:
            message = input("you> ").strip()
        except EOFError:
            break
        if not message:
            continue
        if message.lower() in {"/exit", "/quit"}:
            break
        result = model.generate.remote(
            message=message,
            history=history,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=SEED + len(history),
        )
        response = result["response"]
        print(f"tam> {response}")
        history.extend(
            [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response},
            ]
        )
