import json
import uuid
import time
import asyncio
import re
import hashlib
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from curl_cffi import requests

# 零宽字符定义
ZW_BINARY_0 = '\u200b'
ZW_BINARY_1 = '\u200c'
ZW_START = '\u200d\u200b'
ZW_END = '\u200d\u200c'
ZW_SEP = '\u200d\u200d'

def get_content_hash(text: str) -> str:
    """计算文本内容的简短 Hash (取 SHA256 前 8 位)"""
    # 先去除文本中可能存在的旧 Flag 再计算 Hash
    clean_text = strip_flag(text)
    return hashlib.sha256(clean_text.encode('utf-8')).hexdigest()[:8]

def strip_flag(text: str) -> str:
    """从文本中完全移除零宽 Flag 字符"""
    pattern = f"{ZW_START}[\\{ZW_BINARY_0}\\{ZW_BINARY_1}]+{ZW_END}"
    return re.sub(pattern, "", text)

def encode_flag(context_id: str, content_hash: str = "") -> str:
    """将 ID 和 Hash 编码为零宽字符 Flag"""
    payload = f"{context_id}|{content_hash}"
    binary = "".join(format(b, '08b') for b in payload.encode('utf-8'))
    zw_payload = "".join(ZW_BINARY_0 if b == '0' else ZW_BINARY_1 for b in binary)
    return f"{ZW_START}{zw_payload}{ZW_END}"

def decode_flag(text: str) -> Optional[Dict[str, str]]:
    """从文本末尾解码零宽字符 Flag"""
    pattern = f"{ZW_START}([\\{ZW_BINARY_0}\\{ZW_BINARY_1}]+){ZW_END}"
    match = re.search(pattern, text)
    if not match:
        return None
    
    zw_payload = match.group(1)
    binary = "".join('0' if c == ZW_BINARY_0 else '1' for c in zw_payload)
    
    try:
        byte_data = bytearray()
        for i in range(0, len(binary), 8):
            byte_data.append(int(binary[i:i+8], 2))
        
        decoded = byte_data.decode('utf-8')
        parts = decoded.split('|')
        return {
            "context_id": parts[0],
            "hash": parts[1] if len(parts) > 1 else ""
        }
    except Exception:
        return None

app = FastAPI(title="ChatSDK OpenAI Wrapper")

# 上下文映射表: context_id -> {"chat_id": str, "history": list}
context_store: Dict[str, Dict[str, Any]] = {}

class ChatSDKClient:
    def __init__(self):
        self.base_url = "https://demo.chat-sdk.dev"
        self.curl_session = requests.AsyncSession(impersonate="chrome110")
        self.csrf_token = None
        self.user_data = None

    async def initialize(self):
        print("[*] Refreshing Session (Async)...")
        # 清除旧的 Session 状态
        self.curl_session = requests.AsyncSession(impersonate="chrome110")
        self.curl_session.headers.update({
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
        })
        
        # 1. Get CSRF
        resp = await self.curl_session.get(f"{self.base_url}/api/auth/csrf")
        self.csrf_token = resp.json().get("csrfToken")

        # 2. Trigger guest session
        await self.curl_session.post(
            f"{self.base_url}/api/auth/callback/guest",
            data={
                "csrfToken": self.csrf_token,
                "callbackUrl": self.base_url,
                "json": "true"
            }
        )

        # 3. Get Session
        resp = await self.curl_session.get(f"{self.base_url}/api/auth/session")
        self.user_data = resp.json()
        
        if not self.user_data or not self.user_data.get("user"):
             await self.curl_session.get(f"{self.base_url}/")
             resp = await self.curl_session.get(f"{self.base_url}/api/auth/session")
             self.user_data = resp.json()
        
        return self.user_data

    async def chat_stream(self, messages: List[Dict[str, Any]], model: str):
        if not self.csrf_token:
            await self.initialize()

        # 检测上下文复用
        chat_id = str(uuid.uuid4())
        history = []
        last_msg_content = messages[-1]["content"]
        
        # 检查最后一条助理消息是否包含有效 Flag 且内容未被篡改
        flag_info = None
        assistant_msg_index = -1
        if len(messages) > 1:
            for i in range(len(messages) - 2, -1, -1):
                if messages[i]["role"] == "assistant":
                    flag_info = decode_flag(messages[i]["content"])
                    assistant_msg_index = i
                    break
        
        should_reuse = False
        if flag_info:
            ctx_id = flag_info["context_id"]
            stored_hash = flag_info["hash"]
            # 计算当前消息内容的 Hash (去除 Flag 后)
            current_content = messages[assistant_msg_index]["content"]
            actual_hash = get_content_hash(current_content)
            
            if ctx_id in context_store and actual_hash == stored_hash:
                print(f"[*] 检测到FLAG且Hash校验通过，复用上下文 <{ctx_id}>")
                stored = context_store[ctx_id]
                chat_id = stored["chat_id"]
                history = stored["history"]
                should_reuse = True
            elif actual_hash != stored_hash:
                print(f"[*] 检测到内容篡改 (Hash: {actual_hash} != {stored_hash})，复用失败，以降级后的内容重建上下文")
            else:
                print(f"[*] 复用失败 (ContextID 不存在)，降级并重建上下文")

        if not should_reuse:
            # 降级逻辑：手动重新构建上下文，并清洗所有消息中的 Flag
            for msg in messages[:-1]:
                clean_content = strip_flag(msg["content"])
                history.append({
                    "id": str(uuid.uuid4()),
                    "role": msg["role"],
                    "parts": [{"type": "text", "text": clean_content}]
                })

        # 清洗最后一条用户消息
        last_message = {
            "id": str(uuid.uuid4()),
            "role": messages[-1]["role"],
            "parts": [{"type": "text", "text": strip_flag(last_msg_content)}]
        }

        # 如果 history 不为空，将 Prompt 注入到 history[0] 的开头
        if history:
            original_text = history[0]["parts"][0]["text"]
            prompt = "You are a helpful assistant." # 默认 Prompt 或从某处获取
            # 如果 messages[0] 是 system 角色，则使用它的内容作为 prompt
            if messages[0]["role"] == "system":
                prompt = messages[0]["content"]
            
            history[0]["parts"][0]["text"] = f"<SYSTEM_PROMPT>\n{prompt}\n</SYSTEM_PROMPT>\n\n{original_text}"
        else:
            # 如果没有 history，则注入到 last_message 的开头
            original_text = last_message["parts"][0]["text"]
            prompt = "You are a helpful assistant."
            if messages[0]["role"] == "system":
                prompt = messages[0]["content"]
            
            last_message["parts"][0]["text"] = f"<SYSTEM_PROMPT>\n{prompt}\n</SYSTEM_PROMPT>\n\n{original_text}"

        url = f"{self.base_url}/api/chat"
        headers = {
            "accept": "*/*",
            "content-type": "application/json",
            "origin": self.base_url,
            "referer": f"{self.base_url}/chat/{chat_id}",
            "x-csrf-token": self.csrf_token,
        }
        
        payload = {
            "id": chat_id,
            "selectedChatModel": model,
            "selectedVisibilityType": "private",
            "messages": history + [last_message]
        }
        
        # 重试机制
        attempts = 0
        while attempts < 2:
            print(f"[*] Sending Chat Request (Attempt {attempts + 1}, Async)...")
            headers["x-csrf-token"] = self.csrf_token 
            
            full_response_text = ""
            try:
                resp = await self.curl_session.post(url, headers=headers, json=payload, stream=True)
                
                if resp.status_code == 200:
                    # 状态追踪
                    last_code_content = ""
                    in_code_block = False
                    current_tool_call = {}

                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        line_str = line.decode('utf-8')
                        if line_str.startswith("data: "):
                            data_content = line_str[6:]
                            if data_content == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_content)
                                chunk_type = chunk.get("type")
                                
                                delta_text = ""
                                
                                # 1. 处理标准文本
                                if chunk_type == "text-delta":
                                    if in_code_block:
                                        delta_text = "\n```\n" + chunk.get("delta", "")
                                        in_code_block = False
                                    else:
                                        delta_text = chunk.get("delta", "")
                                
                                # 2. 处理工具输入 (JSON 形式的参数)
                                elif chunk_type == "tool-input-delta":
                                    delta_text = chunk.get("inputTextDelta", "")
                                
                                # 3. 处理代码生成 (累积形式的内容)
                                elif chunk_type == "data-codeDelta":
                                    full_content = chunk.get("data", "")
                                    if not in_code_block:
                                        # 尝试获取语言类型，默认为 python
                                        lang = current_tool_call.get("kind", "python") if current_tool_call else "python"
                                        title = current_tool_call.get("title", "") if current_tool_call else ""
                                        header = f"\n\n### {title}\n" if title else "\n"
                                        delta_text = f"{header}```{lang}\n"
                                        in_code_block = True
                                    
                                    # 计算增量 (diff)
                                    if full_content.startswith(last_code_content):
                                        delta_text += full_content[len(last_code_content):]
                                    else:
                                        delta_text += full_content
                                    
                                    last_code_content = full_content
                                
                                # 4. 处理元数据
                                elif chunk_type == "data-kind":
                                    if not current_tool_call: current_tool_call = {}
                                    current_tool_call["kind"] = chunk.get("data")
                                
                                elif chunk_type == "data-title":
                                    if not current_tool_call: current_tool_call = {}
                                    current_tool_call["title"] = chunk.get("data")
                                
                                # 5. 处理其他文本增量
                                elif chunk_type == "data-textDelta":
                                    delta_text = chunk.get("data", "")
                                
                                # 6. 处理步骤结束 (关闭代码块)
                                elif chunk_type in ["finish-step", "tool-output-available", "data-finish"]:
                                    if in_code_block:
                                        delta_text = "\n```\n"
                                        in_code_block = False
                                        last_code_content = ""
                                        current_tool_call = {} # 重置工具调用状态
                                
                                if delta_text:
                                    full_response_text += delta_text
                                    yield delta_text
                            except:
                                pass
                    
                    # 保存上下文映射
                    new_ctx_id = str(uuid.uuid4())[:8]
                    # 保存到历史记录时，确保是不含 Flag 的纯文本
                    new_history = history + [last_message, {
                        "id": str(uuid.uuid4()),
                        "role": "assistant",
                        "parts": [{"type": "text", "text": full_response_text}]
                    }]
                    context_store[new_ctx_id] = {
                        "chat_id": chat_id,
                        "history": new_history
                    }
                    
                    # 计算新内容的 Hash 并生成 Flag
                    new_hash = get_content_hash(full_response_text)
                    yield encode_flag(new_ctx_id, new_hash)
                    return 
                else:
                    print(f"[!] Request failed with status {resp.status_code}. Refreshing session...")
                    await self.initialize()
                    attempts += 1
            except Exception as e:
                print(f"[!] Error during request: {e}")
                await self.initialize()
                attempts += 1
        
        yield f"Error: Request failed after {attempts} attempts."

# OpenAI 兼容的模型定义
SUPPORTED_MODELS = [
    "anthropic/claude-opus-4.5",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-haiku-4.5",
    "openai/gpt-4.1-mini",
    "openai/gpt-5.2",
    "google/gemini-2.5-flash-lite",
    "google/gemini-3-pro-preview",
    "xai/grok-4.1-fast-non-reasoning",
    "anthropic/claude-3.7-sonnet-thinking",
    "xai/grok-code-fast-1-thinking"
]

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "google/gemini-3-pro-preview"
    messages: List[Message]
    stream: bool = False

client = ChatSDKClient()

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    messages_dict = [m.model_dump() for m in request.messages]
    
    if request.stream:
        async def generate():
            created_time = int(time.time())
            request_id = f"chatcmpl-{uuid.uuid4()}"
            
            # 发送首个 chunk (role)
            yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': request.model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
            
            async for delta in client.chat_stream(messages_dict, request.model):
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": delta},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            
            # 发送结束 chunk
            yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': request.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        # 非流式处理
        content = ""
        async for delta in client.chat_stream(messages_dict, request.model):
            content += delta
        
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "chatsdk"
            } for model_id in SUPPORTED_MODELS
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
