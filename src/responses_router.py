"""
CodeBuddy Responses API Router - OpenAI Responses API (/v1/responses) 兼容层

将 OpenAI Responses API 请求转换为上游 CodeBuddy Chat Completions 请求，
并把上游 Chat Completions 的 SSE 流实时翻译为 Responses API 的 SSE 事件流，
使官方 ChatGPT.app（内置 Codex，仅支持 wire_api="responses"）可以直接接入本网关。
"""
import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from .auth import authenticate
from .codebuddy_api_client import codebuddy_api_client
from .codebuddy_router import (
    CodeBuddyStreamService,
    CredentialManager,
    RequestProcessor,
    SSE_HEADERS,
    get_codebuddy_api_url,
    get_http_client,
    parse_sse_line,
)
from .usage_stats_manager import usage_stats_manager

logger = logging.getLogger(__name__)

router = APIRouter()

TEXT_BLOCK_TYPES = ("input_text", "output_text", "text")


def _new_id(prefix: str) -> str:
    """生成带前缀的唯一 ID（resp_ / msg_ / fc_ / call_）"""
    return f"{prefix}_{uuid.uuid4().hex}"


def _convert_tool_call_id(call_id: Any) -> str:
    """把上游工具调用 ID 转为 Responses 客户端可回传的 call_id"""
    if call_id is None:
        return _new_id("call")
    call_id = str(call_id)
    if call_id.startswith("tooluse_"):
        return f"call_{call_id[len('tooluse_'):]}"
    return call_id


def _sse(obj: Dict[str, Any]) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _response_base(rid: str, created_at: int, model: str) -> Dict[str, Any]:
    """Responses API response 对象的基础骨架"""
    return {
        "id": rid,
        "object": "response",
        "created_at": created_at,
        "status": "in_progress",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "model": model,
        "output": [],
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": {"effort": None, "summary": None},
        "store": False,
        "temperature": None,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "truncation": None,
        "usage": None,
        "user": None,
        "metadata": {},
    }


def _map_usage(chat_usage: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """把 Chat Completions usage 映射为 Responses usage"""
    if not isinstance(chat_usage, dict):
        return None
    input_tokens = chat_usage.get("prompt_tokens", 0)
    output_tokens = chat_usage.get("completion_tokens", 0)
    reasoning_tokens = 0
    completion_details = chat_usage.get("completion_tokens_details")
    if isinstance(completion_details, dict):
        reasoning_tokens = completion_details.get("reasoning_tokens", 0)
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {
            "cached_tokens": chat_usage.get("cached_tokens", 0),
        },
        "output_tokens": output_tokens,
        "output_tokens_details": {
            "reasoning_tokens": reasoning_tokens,
        },
        "total_tokens": chat_usage.get("total_tokens", input_tokens + output_tokens),
    }


class ResponsesRequestConverter:
    """OpenAI Responses API 请求 → 上游 Chat Completions 请求"""

    @staticmethod
    def _content_blocks_to_text(content: Any) -> str:
        """把 Responses content blocks（或字符串）摊平成纯文本"""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    btype = block.get("type", "")
                    if btype in TEXT_BLOCK_TYPES:
                        parts.append(str(block.get("text", "")))
                    elif btype in ("input_image", "output_image"):
                        logger.info(
                            "[Responses] Ignoring %s content block (image not supported by upstream)", btype
                        )
                    else:
                        text = block.get("text")
                        if text is not None:
                            parts.append(str(text))
            return "".join(parts)
        return str(content)

    @classmethod
    def to_chat_body(cls, body: Dict[str, Any]) -> Dict[str, Any]:
        """把 Responses API 请求体转换为 Chat Completions 请求体"""
        messages: List[Dict[str, Any]] = []

        instructions = body.get("instructions")
        if isinstance(instructions, str) and instructions.strip():
            messages.append({"role": "system", "content": instructions})

        raw_input = body.get("input", [])
        if isinstance(raw_input, str):
            raw_input = [{"type": "message", "role": "user", "content": raw_input}]
        if raw_input is None:
            raw_input = []

        # 连续出现的 function_call 项合并为同一条 assistant 消息的多个 tool_calls
        pending_tool_calls: List[Dict[str, Any]] = []

        def flush_tool_calls() -> None:
            nonlocal pending_tool_calls
            if pending_tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": pending_tool_calls,
                })
                pending_tool_calls = []

        for item in raw_input:
            if isinstance(item, str):
                flush_tool_calls()
                messages.append({"role": "user", "content": item})
                continue
            if not isinstance(item, dict):
                continue

            item_type = item.get("type")
            if item_type == "message":
                flush_tool_calls()
                role = item.get("role", "user")
                if role == "developer":
                    role = "system"
                messages.append({
                    "role": role,
                    "content": cls._content_blocks_to_text(item.get("content")),
                })
            elif item_type == "function_call":
                pending_tool_calls.append({
                    "id": _convert_tool_call_id(item.get("call_id") or item.get("id")),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments") or "{}",
                    },
                })
            elif item_type == "function_call_output":
                flush_tool_calls()
                output = item.get("output")
                if not isinstance(output, str):
                    output = json.dumps(output, ensure_ascii=False)
                messages.append({
                    "role": "tool",
                    "tool_call_id": _convert_tool_call_id(item.get("call_id") or ""),
                    "content": output,
                })
            elif item_type in (
                "reasoning",
                "reasoning_summary",
                "web_search_call",
                "local_web_search_call",
            ):
                # 不向 Chat 上游透传推理/搜索类历史项
                continue
            else:
                logger.info("[Responses] Ignoring unknown input item type: %s", item_type)

        flush_tool_calls()

        if not messages:
            raise HTTPException(
                status_code=400,
                detail="input is required and must not be empty",
            )

        chat_body: Dict[str, Any] = {
            "model": body.get("model") or "auto-chat",
            "messages": messages,
        }

        if body.get("temperature") is not None:
            chat_body["temperature"] = body["temperature"]
        if body.get("top_p") is not None:
            chat_body["top_p"] = body["top_p"]
        max_output_tokens = body.get("max_output_tokens")
        if max_output_tokens not in (None, 0):
            chat_body["max_tokens"] = max_output_tokens

        # tools: 只保留 function 类型；Responses 特有类型（web_search 等）丢弃
        chat_tools: List[Dict[str, Any]] = []
        for tool in body.get("tools") or []:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") == "function":
                fn = {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters")
                    or {"type": "object", "properties": {}},
                }
                if tool.get("strict") is not None:
                    fn["strict"] = tool["strict"]
                chat_tools.append({"type": "function", "function": fn})
            else:
                logger.info(
                    "[Responses] Dropping unsupported tool type for upstream: %s", tool.get("type")
                )
        if chat_tools:
            chat_body["tools"] = chat_tools

        if body.get("tool_choice") is not None:
            tool_choice = body["tool_choice"]
            # Responses 专用形式 {"type":"function","name":"x"} → Chat 形式
            if (
                isinstance(tool_choice, dict)
                and tool_choice.get("type") == "function"
                and "name" in tool_choice
                and "function" not in tool_choice
            ):
                tool_choice = {"type": "function", "function": {"name": tool_choice["name"]}}
            chat_body["tool_choice"] = tool_choice

        if body.get("parallel_tool_calls") is not None:
            chat_body["parallel_tool_calls"] = bool(body["parallel_tool_calls"])

        return chat_body


class ChatToResponsesTranslator:
    """把聚合后的 Chat Completions 响应（非流式）转成 Responses API JSON"""

    @staticmethod
    def build_response(
        chat_completion: Dict[str, Any],
        requested_model: str,
    ) -> Dict[str, Any]:
        response = _response_base(_new_id("resp"), int(time.time()), chat_completion.get("model") or requested_model)
        response["status"] = "completed"

        output: List[Dict[str, Any]] = []
        choices = chat_completion.get("choices") or []
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message") or {}
            text = message.get("content") or ""
            if not isinstance(text, str):
                text = str(text)
            tool_calls = message.get("tool_calls") or []

            if text or not tool_calls:
                output.append({
                    "id": _new_id("msg"),
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                })

            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                output.append({
                    "id": _new_id("fc"),
                    "type": "function_call",
                    "status": "completed",
                    "call_id": _convert_tool_call_id(tc.get("id")),
                    "name": fn.get("name", ""),
                    "arguments": fn.get("arguments") or "{}",
                })

        response["output"] = output
        response["usage"] = _map_usage(chat_completion.get("usage"))
        return response


class ResponsesSSEStreamTranslator:
    """把上游 Chat Completions SSE chunk 实时转成 Responses API SSE 事件帧"""

    def __init__(self, requested_model: str):
        self.response_id = _new_id("resp")
        self.created_at = int(time.time())
        self.model = requested_model
        self._response = _response_base(self.response_id, self.created_at, self.model)

        self._next_output_index = 0

        # 文本 message item 状态（懒创建）
        self._msg_item_id: Optional[str] = None
        self._msg_output_index: Optional[int] = None
        self._msg_text = ""

        # 工具调用状态（按到达顺序）
        self._tool_states: List[Dict[str, Any]] = []
        self._by_original_id: Dict[str, Dict[str, Any]] = {}
        self._by_chat_index: Dict[int, Dict[str, Any]] = {}

        self.usage: Optional[Dict[str, Any]] = None

    # ---------- 内部工具 ----------

    def _take_output_index(self) -> int:
        idx = self._next_output_index
        self._next_output_index += 1
        return idx

    def _ensure_message_item(self) -> List[str]:
        """首次出现文本时创建 message output item，返回需要补发的 SSE 帧"""
        if self._msg_item_id is not None:
            return []
        self._msg_item_id = _new_id("msg")
        self._msg_output_index = self._take_output_index()
        item = {
            "id": self._msg_item_id,
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        }
        part = {"type": "output_text", "text": "", "annotations": []}
        return [
            _sse({
                "type": "response.output_item.added",
                "output_index": self._msg_output_index,
                "item": item,
            }),
            _sse({
                "type": "response.content_part.added",
                "item_id": self._msg_item_id,
                "output_index": self._msg_output_index,
                "content_index": 0,
                "part": part,
            }),
        ]

    def _create_tool_state(self, original_id: str, chat_index: Optional[Any]) -> Dict[str, Any]:
        state = {
            "original_id": original_id,
            "call_id": _convert_tool_call_id(original_id),
            "item_id": _new_id("fc"),
            "output_index": self._take_output_index(),
            "name": "",
            "args": "",
            "added_sent": False,
        }
        self._tool_states.append(state)
        if chat_index is not None:
            self._by_chat_index[int(chat_index)] = state
        return state

    def _handle_content(self, text: Any) -> List[str]:
        if text is None:
            return []
        text = str(text)
        if not text:
            return []
        self._msg_text += text
        frames = self._ensure_message_item()
        frames.append(_sse({
            "type": "response.output_text.delta",
            "item_id": self._msg_item_id,
            "output_index": self._msg_output_index,
            "content_index": 0,
            "delta": text,
        }))
        return frames

    def _handle_tool_call(self, tc: Dict[str, Any]) -> List[str]:
        original_id = tc.get("id")
        chat_index = tc.get("index")
        func = tc.get("function") or {}
        name = func.get("name")
        args = func.get("arguments") or ""

        state: Optional[Dict[str, Any]] = None
        if original_id is not None and str(original_id) in self._by_original_id:
            state = self._by_original_id[str(original_id)]
        elif chat_index is not None and int(chat_index) in self._by_chat_index and original_id is None:
            state = self._by_chat_index[int(chat_index)]
        elif original_id is not None:
            state = self._create_tool_state(str(original_id), chat_index)
            self._by_original_id[str(original_id)] = state
        elif chat_index is not None and 0 <= int(chat_index) < len(self._tool_states):
            # 分片只带 index 时，按工具调用到达顺序兜底匹配
            state = self._tool_states[int(chat_index)]
            self._by_chat_index[int(chat_index)] = state
        elif chat_index is not None and not self._tool_states:
            # 极少数上游首个分片可能不带 id，兜底生成
            state = self._create_tool_state(f"gen_{chat_index}", chat_index)
        elif self._tool_states:
            state = self._tool_states[-1]

        if state is None:
            return []

        if name and not state["name"]:
            state["name"] = str(name)
        if args:
            state["args"] += str(args)

        frames: List[str] = []
        if not state["added_sent"]:
            state["added_sent"] = True
            item = {
                "id": state["item_id"],
                "type": "function_call",
                "status": "in_progress",
                "call_id": state["call_id"],
                "name": state["name"],
                "arguments": "",
                "parallel_tool_calls": True,
            }
            frames.append(_sse({
                "type": "response.output_item.added",
                "output_index": state["output_index"],
                "item": item,
            }))
        if args:
            frames.append(_sse({
                "type": "response.function_call_arguments.delta",
                "item_id": state["item_id"],
                "output_index": state["output_index"],
                "delta": str(args),
            }))
        return frames

    # ---------- 对外接口 ----------

    def start_frames(self) -> List[str]:
        return [
            _sse({"type": "response.created", "response": self._response}),
            _sse({"type": "response.in_progress", "response": self._response}),
        ]

    def process_chunk(self, chunk: Dict[str, Any]) -> List[str]:
        """处理一个上游 Chat Completions chunk，返回待发送的 Responses SSE 帧"""
        frames: List[str] = []
        if not isinstance(chunk, dict):
            return frames

        if chunk.get("error"):
            frames.append(_sse({
                "type": "error",
                "code": (chunk["error"].get("code") if isinstance(chunk["error"], dict) else None) or "upstream_error",
                "message": (chunk["error"].get("message") if isinstance(chunk["error"], dict) else str(chunk["error"]))
                or "upstream error",
                "param": None,
            }))
            return frames

        if chunk.get("model"):
            self.model = chunk["model"]
            self._response["model"] = self.model
        if chunk.get("usage"):
            self.usage = chunk["usage"]

        choices = chunk.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            return frames

        choice = choices[0]
        if choice.get("finish_reason"):
            self._response["finish_reason"] = choice["finish_reason"]

        delta = choice.get("delta") or {}
        if delta.get("content"):
            frames += self._handle_content(delta["content"])
        for tc in delta.get("tool_calls") or []:
            if isinstance(tc, dict):
                frames += self._handle_tool_call(tc)
        return frames

    def finish_frames(self) -> List[str]:
        """流结束时补齐 done / completed 事件"""
        frames: List[str] = []
        output: List[Dict[str, Any]] = []

        if self._msg_item_id is not None:
            content_part = {"type": "output_text", "text": self._msg_text, "annotations": []}
            frames.append(_sse({
                "type": "response.output_text.done",
                "item_id": self._msg_item_id,
                "output_index": self._msg_output_index,
                "content_index": 0,
                "text": self._msg_text,
            }))
            frames.append(_sse({
                "type": "response.content_part.done",
                "item_id": self._msg_item_id,
                "output_index": self._msg_output_index,
                "content_index": 0,
                "part": content_part,
            }))
            msg_item = {
                "id": self._msg_item_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [content_part],
            }
            frames.append(_sse({
                "type": "response.output_item.done",
                "output_index": self._msg_output_index,
                "item": msg_item,
            }))
            output.append(msg_item)

        for state in self._tool_states:
            frames.append(_sse({
                "type": "response.function_call_arguments.done",
                "item_id": state["item_id"],
                "output_index": state["output_index"],
                "arguments": state["args"],
            }))
            item = {
                "id": state["item_id"],
                "type": "function_call",
                "status": "completed",
                "call_id": state["call_id"],
                "name": state["name"],
                "arguments": state["args"],
            }
            frames.append(_sse({
                "type": "response.output_item.done",
                "output_index": state["output_index"],
                "item": item,
            }))
            output.append(item)

        self._response["status"] = "completed"
        self._response["output"] = output
        self._response["usage"] = _map_usage(self.usage)
        frames.append(_sse({"type": "response.completed", "response": self._response}))
        return frames

    def error_frames(self, message: str, code: str = "upstream_error") -> List[str]:
        self._response["status"] = "failed"
        self._response["error"] = {"code": code, "message": message}
        return [
            _sse({"type": "error", "code": code, "message": message, "param": None}),
            _sse({"type": "response.failed", "response": self._response}),
        ]


async def _stream_responses(payload: Dict[str, Any], headers: Dict[str, str]) -> AsyncGenerator[str, None]:
    """把上游 Chat SSE 流转成 Responses SSE 事件流"""
    translator = ResponsesSSEStreamTranslator(payload.get("model") or "auto-chat")
    for frame in translator.start_frames():
        yield frame

    buffer = ""
    try:
        client = await get_http_client()
        async with client.stream(
            "POST", get_codebuddy_api_url(), json=payload, headers=headers
        ) as response:
            if response.status_code != 200:
                error_text = await response.aread()
                error_msg = error_text.decode("utf-8", errors="ignore")
                logger.error("CodeBuddy Responses upstream error: %s - %s", response.status_code, error_msg[:500])
                for frame in translator.error_frames(
                    f"CodeBuddy API error: {response.status_code} - {error_msg[:500]}"
                ):
                    yield frame
                return

            async for text_chunk in response.aiter_text(chunk_size=8192):
                if not text_chunk:
                    continue
                buffer += text_chunk
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if not line.strip() or line.startswith(":"):
                        continue
                    if "[DONE]" in line:
                        for frame in translator.finish_frames():
                            yield frame
                        return
                    chunk_data = parse_sse_line(line)
                    if chunk_data:
                        for frame in translator.process_chunk(chunk_data):
                            yield frame

            if buffer.strip():
                chunk_data = parse_sse_line(buffer.strip())
                if chunk_data:
                    for frame in translator.process_chunk(chunk_data):
                        yield frame

            for frame in translator.finish_frames():
                yield frame
    except Exception as e:  # noqa: BLE001 - 流式响应开始后无法返回 HTTP 错误，只能发 SSE 错误事件
        logger.error("Responses stream error: %s", e)
        for frame in translator.error_frames(f"Stream error: {str(e)}"):
            yield frame


@router.post("/v1/responses")
async def create_response(
    request: Request,
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_conversation_request_id: Optional[str] = Header(None, alias="X-Conversation-Request-ID"),
    x_conversation_message_id: Optional[str] = Header(None, alias="X-Conversation-Message-ID"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
    _token: str = Depends(authenticate),
):
    """OpenAI Responses API 兼容端点（供 Codex / ChatGPT.app 使用）"""
    try:
        try:
            request_body = await request.json()
        except Exception as e:
            logger.error("解析 Responses 请求体失败: %s", e)
            raise HTTPException(status_code=400, detail=f"Invalid JSON request body: {str(e)}")

        if not isinstance(request_body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object")

        # Responses 请求 → Chat Completions 请求
        chat_body = ResponsesRequestConverter.to_chat_body(request_body)
        RequestProcessor.validate_request(chat_body)

        # 获取认证信息并生成请求头
        auth_context = CredentialManager.get_auth_context()
        headers = codebuddy_api_client.generate_codebuddy_headers(
            auth=auth_context,
            user_id=auth_context.get("user_id"),
            conversation_id=x_conversation_id,
            conversation_request_id=x_conversation_request_id,
            conversation_message_id=x_conversation_message_id,
            request_id=x_request_id,
        )

        payload = RequestProcessor.prepare_payload(chat_body)
        usage_stats_manager.record_model_usage(payload.get("model", "unknown"))
        requested_model = payload.get("model") or "auto-chat"

        if request_body.get("stream", False):
            return StreamingResponse(
                _stream_responses(payload, headers),
                media_type="text/event-stream",
                headers=SSE_HEADERS,
            )

        # 非流式：复用现有聚合逻辑（上游只支持流式，网关内部聚合）
        service = CodeBuddyStreamService()
        aggregated = await service.handle_non_stream_response(payload, headers)
        return ChatToResponsesTranslator.build_response(aggregated, requested_model)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("CodeBuddy Responses API 错误: %s", e)
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")
