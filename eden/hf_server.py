"""Optional single-GPU Transformers server for an already installed model/runtime.

This is intentionally separate from the dependency-free simulation. No downloads,
remote code, or runtime installations occur. Use a production server for throughput.
"""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default="qwen-local")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("A Slurm-allocated CUDA GPU is required")
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16,
                                               local_files_only=True, trust_remote_code=False,
                                               attn_implementation="sdpa").to("cuda").eval()
    generation_lock = threading.Lock()
    metadata = {"model_path": args.model_path, "torch": torch.__version__, "transformers": transformers.__version__,
                "gpu": torch.cuda.get_device_name(), "dtype": "bfloat16", "attention": "sdpa"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, data, status=200):
            raw = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/health":
                return self.reply({"ready": True, **metadata})
            if self.path == "/v1/models":
                return self.reply({"data": [{"id": args.model_name, "object": "model"}]})
            self.reply({"error": "Not found"}, 404)

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return self.reply({"error": "Not found"}, 404)
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 2 * 1024 * 1024:
                    raise ValueError("Invalid request size")
                request = json.loads(self.rfile.read(size))
                if request.get("model") != args.model_name or request.get("stream"):
                    raise ValueError("Unknown model or streaming requested")
                if request.get("response_format", {}).get("type") not in {None, "text"}:
                    raise ValueError("Use response_format=none: this server does not implement constrained JSON decoding")
                maximum = int(request.get("max_tokens", 1400))
                temperature = float(request.get("temperature", .6))
                if not 1 <= maximum <= 4096 or not 0 <= temperature <= 2:
                    raise ValueError("Invalid token or temperature budget")
                with generation_lock, torch.inference_mode():
                    prompt = tokenizer.apply_chat_template(request["messages"], tokenize=False, add_generation_prompt=True)
                    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                    prompt_tokens = inputs["input_ids"].shape[1]
                    if prompt_tokens + maximum > min(model.config.max_position_embeddings, 32768):
                        raise ValueError("Context exceeds server's 32768-token budget; reduce context/output")
                    kwargs = {"max_new_tokens": maximum, "do_sample": temperature > 0,
                              "pad_token_id": tokenizer.eos_token_id}
                    if temperature > 0:
                        kwargs.update(temperature=temperature, top_p=.9)
                    start = time.monotonic()
                    generated = model.generate(**inputs, **kwargs)
                    tokens = generated[0, prompt_tokens:]
                    content = tokenizer.decode(tokens, skip_special_tokens=True)
                    output_tokens = len(tokens)
                    elapsed = time.monotonic() - start
                self.reply({"object": "chat.completion", "model": args.model_name,
                            "choices": [{"index": 0, "finish_reason": "length" if output_tokens >= maximum else "stop",
                                         "message": {"role": "assistant", "content": content}}],
                            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": output_tokens,
                                      "total_tokens": prompt_tokens + output_tokens}, "eden_backend": metadata})
                print(json.dumps({"prompt_tokens": prompt_tokens, "output_tokens": output_tokens, "seconds": round(elapsed, 2)}), flush=True)
            except (ValueError, KeyError, TypeError) as exc:
                self.reply({"error": str(exc)}, 400)
            except Exception as exc:
                print(f"Generation error: {type(exc).__name__}: {exc}", flush=True)
                self.reply({"error": f"{type(exc).__name__}: {exc}"}, 500)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(json.dumps({"ready": True, "port": args.port, **metadata}), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
