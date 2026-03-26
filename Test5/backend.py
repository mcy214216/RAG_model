import asyncio
import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

app = FastAPI()

# ---------- 全局加载模型（启动时加载一次） ----------
MODEL_NAME = "model/qwen/Qwen2.5-0.5B-Instruct"  # 可根据需要更换其他 Qwen2.5 模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,          # 使用半精度节省显存
    device_map="auto",                  # 自动分配到可用 GPU/CPU
    trust_remote_code=True
)
model.eval()

# ---------- 请求数据结构 ----------
class ChatRequest(BaseModel):
    query: str
    sys_prompt: str = "You are a helpful assistant."
    history_len: int = 1
    history: List[Dict[str, str]] = []
    temperature: float = 0.5
    top_p: float = 0.5
    max_tokens: int = 1024
    stream: bool = True

# ---------- 构建对话消息 ----------
def build_messages(request: ChatRequest):
    """
    根据请求构建 Qwen 格式的对话消息列表。
    """
    messages = []
    # 系统提示词
    messages.append({"role": "system", "content": request.sys_prompt})
    # 历史对话（保留最近 history_len 条）
    history = request.history[-request.history_len:] if request.history_len > 0 else []
    for msg in history:
        messages.append(msg)
    # 当前用户问题
    messages.append({"role": "user", "content": request.query})
    return messages

# ---------- 非流式生成 ----------
def generate_non_stream(request: ChatRequest):
    messages = build_messages(request)
    # Qwen 的 chat 模板（如果模型有 apply_chat_template 则使用）
    if hasattr(tokenizer, 'apply_chat_template'):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        # 简易拼接（不一定通用，建议使用 apply_chat_template）
        text = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant: "
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response_text

# ---------- 流式生成 ----------
async def generate_stream(request: ChatRequest):
    messages = build_messages(request)
    if hasattr(tokenizer, 'apply_chat_template'):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        text = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant: "
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # 使用 TextIteratorStreamer 实现流式
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        inputs,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )
    # 在后台线程运行生成
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 逐步输出生成的文本块
    for new_text in streamer:
        yield new_text
        await asyncio.sleep(0)  # 让出控制权，保持异步

# ---------- 路由 ----------
@app.post("/chat")
async def chat(request: ChatRequest):
    if request.stream:
        return StreamingResponse(generate_stream(request), media_type="text/plain")
    else:
        response = generate_non_stream(request)
        return response

# ---------- 健康检查（可选） ----------
@app.get("/")
def root():
    return {"status": "ok", "model": MODEL_NAME}