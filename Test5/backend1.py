# -*- coding: utf-8 -*-
# @Time    : 2026/3/26 10:49
# @Author  : mcy
# @File    : backend1.py
import asyncio
import requests
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional

app = FastAPI()

# Ollama 服务地址（默认端口 11434）
OLLAMA_URL = "http://localhost:11434/api/chat"

# 默认模型名称（可修改为你拉取的模型，如 qwen2.5:7b）
DEFAULT_MODEL = "gemma3:4b"

# 请求数据结构（与前端一致）
class ChatRequest(BaseModel):
    query: str
    sys_prompt: str = "You are a helpful assistant."
    history_len: int = 1
    history: List[Dict[str, str]] = []
    temperature: float = 0.5
    top_p: float = 0.5
    max_tokens: int = 1024
    stream: bool = True

def build_ollama_messages(request: ChatRequest):
    """
    将前端传来的消息历史转换为 Ollama 的 messages 格式。
    Ollama 的消息格式：[{"role": "system", "content": ...}, {"role": "user", "content": ...}, ...]
    """
    messages = []
    # 添加系统提示
    if request.sys_prompt:
        messages.append({"role": "system", "content": request.sys_prompt})
    # 添加历史对话（保留最近 history_len 条）
    # 注意：前端 history 中包含 user 和 assistant 交替的消息，我们直接使用
    history = request.history[-request.history_len:] if request.history_len > 0 else []
    messages.extend(history)
    # 添加当前用户消息
    messages.append({"role": "user", "content": request.query})
    return messages

def generate_non_stream(request: ChatRequest):
    """非流式调用 Ollama"""
    messages = build_ollama_messages(request)
    payload = {
        "model": DEFAULT_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "num_predict": request.max_tokens  # Ollama 中使用 num_predict 控制生成长度
        }
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    # Ollama 返回的格式：{"message": {"content": "..."}, ...}
    return data.get("message", {}).get("content", "")

async def generate_stream(request: ChatRequest):
    """流式调用 Ollama"""
    messages = build_ollama_messages(request)
    payload = {
        "model": DEFAULT_MODEL,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "num_predict": request.max_tokens
        }
    }
    # 发送流式请求
    with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=60) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                # Ollama 每行返回一个 JSON 对象
                try:
                    chunk = json.loads(line)
                    if "message" in chunk and "content" in chunk["message"]:
                        content = chunk["message"]["content"]
                        yield content
                except json.JSONDecodeError:
                    continue
        # 可选：最后可以加一个结束标记，但不需要
        await asyncio.sleep(0)

@app.post("/chat")
async def chat(request: ChatRequest):
    if request.stream:
        return StreamingResponse(generate_stream(request), media_type="text/plain")
    else:
        response_text = generate_non_stream(request)
        return response_text

@app.get("/")
def root():
    return {"status": "ok", "model": DEFAULT_MODEL, "backend": "ollama"}