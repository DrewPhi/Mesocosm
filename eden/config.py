import dataclasses
import json
import math
from pathlib import Path
from urllib.parse import urlparse


@dataclasses.dataclass
class Config:
    seed: int = 104729
    endpoint: str = "http://127.0.0.1:8000/v1"
    model: str = ""
    referee_model: str | None = None
    api_key_env: str = "EDEN_API_KEY"
    backend: str = "openai"
    response_format: str = "json_object"
    timeout: float = 120
    retries: int = 2
    max_tokens: int = 1400
    temperature: float = 0.6
    referee_temperature: float = 0.1
    speed: float = 0
    max_population: int = 100
    activation_seconds: int = 30
    wake_interval: int = 1800
    initial_age_years: int = 22
    lifespan_years: int = 75

    def validate(self):
        if self.backend not in {"mock", "openai"}:
            raise ValueError("backend must be mock or openai")
        if self.backend == "openai" and not self.model:
            raise ValueError("Set --model to the model name exposed by your inference server")
        url = urlparse(self.endpoint)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query:
            raise ValueError("endpoint must be an HTTP(S) URL without credentials or query parameters")
        if self.response_format not in {"json_object", "none"}:
            raise ValueError("response_format must be json_object or none")
        for key in ("timeout", "max_tokens", "max_population", "activation_seconds", "wake_interval", "lifespan_years"):
            value = getattr(self, key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be positive and finite")
        if not 0 <= self.retries <= 5 or not math.isfinite(self.speed) or self.speed < 0:
            raise ValueError("retries must be 0–5; speed must be finite and nonnegative (0 = max)")
        if self.initial_age_years < 18:
            raise ValueError("Founders must be adults")
        for key in ("temperature", "referee_temperature"):
            value = getattr(self, key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 2:
                raise ValueError(f"{key} must be between 0 and 2")
        for key in ("seed", "retries", "max_tokens", "max_population", "activation_seconds", "wake_interval", "initial_age_years", "lifespan_years"):
            if type(getattr(self, key)) is not int:
                raise ValueError(f"{key} must be an integer")
        return self

    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text())) if path else cls()
