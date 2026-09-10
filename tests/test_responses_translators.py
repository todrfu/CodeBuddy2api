"""
离线单测：Responses API ↔ Chat Completions 翻译层
运行: /tmp/cb2api-venv/bin/python tests/test_responses_translators.py
也兼容 pytest（如已安装）。
"""
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.responses_router import (  # noqa: E402
    ChatToResponsesTranslator,
    ResponsesRequestConverter,
    ResponsesSSEStreamTranslator,
)


def _parse_frames(frames):
    """把 SSE 帧字符串解析为 (type, payload) 列表"""
    out = []
    for frame in frames:
        line = frame.strip()
        assert line.startswith("data: "), f"bad sse frame: {frame!r}"
        payload = json.loads(line[len("data: "):])
        out.append(payload)
    return out


def test_request_text_basic():
    body = {
        "model": "glm-5.3",
        "instructions": "Be brief.",
        "input": [
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "hi"}
            ]}
        ],
        "max_output_tokens": 100,
        "temperature": 0.2,
    }
    chat = ResponsesRequestConverter.to_chat_body(body)
    assert chat["model"] == "glm-5.3"
    assert chat["max_tokens"] == 100
    assert chat["temperature"] == 0.2
    roles = [m["role"] for m in chat["messages"]]
    assert roles == ["system", "user"]
    assert chat["messages"][0]["content"] == "Be brief."
    assert chat["messages"][1]["content"] == "hi"


def test_request_string_input():
    chat = ResponsesRequestConverter.to_chat_body({"input": "hello"})
    assert chat["messages"][0]["role"] == "user"
    assert chat["messages"][0]["content"] == "hello"


def test_request_tool_history_and_tools():
    body = {
        "model": "glm-5.3",
        "input": [
            {"type": "message", "role": "user", "content": "weather?"},
            {
                "type": "function_call",
                "call_id": "tooluse_abc123",
                "name": "get_weather",
                "arguments": '{"city": "SZ"}',
            },
            {
                "type": "function_call_output",
                "call_id": "tooluse_abc123",
                "output": "sunny",
            },
        ],
        "tools": [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                "strict": True,
            },
            {"type": "web_search"},
        ],
        "tool_choice": {"type": "function", "name": "get_weather"},
    }
    chat = ResponsesRequestConverter.to_chat_body(body)

    # 历史轮次: user -> assistant(tool_calls) -> tool
    assert [m["role"] for m in chat["messages"]] == ["user", "assistant", "tool"]
    assistant = chat["messages"][1]
    assert assistant["tool_calls"][0]["id"] == "call_abc123"
    assert assistant["tool_calls"][0]["function"]["name"] == "get_weather"
    tool_msg = chat["messages"][2]
    assert tool_msg["tool_call_id"] == "call_abc123"
    assert tool_msg["content"] == "sunny"

    # tools 只保留 function，并带上 strict
    assert len(chat["tools"]) == 1
    assert chat["tools"][0]["type"] == "function"
    assert chat["tools"][0]["function"]["name"] == "get_weather"
    assert chat["tools"][0]["function"]["strict"] is True

    # tool_choice: Responses 形式 → Chat 形式
    assert chat["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}


def _simulate_stream(chunks):
    """模拟上游 Chat SSE: 返回起始帧 + 逐块帧 + 结束帧"""
    translator = ResponsesSSEStreamTranslator("glm-5.3")
    frames = list(translator.start_frames())
    for chunk in chunks:
        frames += translator.process_chunk(chunk)
    frames += translator.finish_frames()
    return frames


def test_stream_text_then_tool_call():
    chunks = [
        {"id": "chatcmpl-1", "model": "glm-5.3", "choices": [
            {"index": 0, "delta": {"role": "assistant", "content": "Hello"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"id": "tooluse_1", "type": "function", "function": {"name": "get_weather", "arguments": ""}}]},
            "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '{"city":"SZ"}'}}]},
            "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
         "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
    ]
    events = _parse_frames(_simulate_stream(chunks))
    types = [e["type"] for e in events]

    assert types[0] == "response.created"
    assert types[1] == "response.in_progress"
    assert "response.output_item.added" in types
    assert "response.output_text.delta" in types
    assert "response.function_call_arguments.delta" in types
    assert types[-1] == "response.completed"

    completed = events[-1]["response"]
    assert completed["status"] == "completed"
    assert completed["id"].startswith("resp_")
    assert completed["model"] == "glm-5.3"
    assert completed["usage"]["input_tokens"] == 10
    assert completed["usage"]["output_tokens"] == 5

    # output[0] = 文本消息, output[1] = function_call
    assert completed["output"][0]["type"] == "message"
    assert completed["output"][0]["content"][0]["text"] == "Hello world"
    assert completed["output"][1]["type"] == "function_call"
    assert completed["output"][1]["call_id"] == "call_1"
    assert completed["output"][1]["name"] == "get_weather"
    assert json.loads(completed["output"][1]["arguments"]) == {"city": "SZ"}

    # 流中的 deltas 与 done 事件顺序正确
    deltas = [e["delta"] for e in events if e["type"] == "response.output_text.delta"]
    assert "".join(deltas) == "Hello world"
    done_types = [t for t in ("response.output_text.done", "response.function_call_arguments.done") if t in types]
    assert len(done_types) == 2


def test_stream_multiple_tool_calls():
    chunks = [
        {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"id": "tooluse_a", "type": "function", "function": {"name": "f_a", "arguments": ""}}]},
            "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"id": "tooluse_b", "type": "function", "function": {"name": "f_b", "arguments": ""}}]},
            "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 1, "function": {"arguments": '{"x":1}'}}]}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '{"y":2}'}}]}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    ]
    completed = _parse_frames(_simulate_stream(chunks))[-1]["response"]
    calls = [o for o in completed["output"] if o["type"] == "function_call"]
    assert len(calls) == 2
    assert calls[0]["call_id"] == "call_a" and json.loads(calls[0]["arguments"]) == {"y": 2}
    assert calls[1]["call_id"] == "call_b" and json.loads(calls[1]["arguments"]) == {"x": 1}


def test_stream_text_only():
    chunks = [
        {"model": "glm-5.3", "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}},
    ]
    completed = _parse_frames(_simulate_stream(chunks))[-1]["response"]
    assert len(completed["output"]) == 1
    assert completed["output"][0]["content"][0]["text"] == "ok"
    assert completed["status"] == "completed"


def test_non_stream_aggregated_to_responses():
    chat_completion = {
        "id": "chatcmpl-x",
        "model": "glm-5.3",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "thinking out loud",
                "tool_calls": [{
                    "id": "tooluse_9",
                    "type": "function",
                    "function": {"name": "apply_patch", "arguments": '{"file":"a.py"}'},
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 8, "completion_tokens": 6, "total_tokens": 14},
    }
    resp = ChatToResponsesTranslator.build_response(chat_completion, "glm-5.3")
    assert resp["object"] == "response"
    assert resp["status"] == "completed"
    assert resp["usage"]["input_tokens"] == 8
    types = [o["type"] for o in resp["output"]]
    assert "message" in types and "function_call" in types
    fc = [o for o in resp["output"] if o["type"] == "function_call"][0]
    assert fc["call_id"] == "call_9"
    assert fc["name"] == "apply_patch"


def test_error_frame():
    translator = ResponsesSSEStreamTranslator("glm-5.3")
    frames = translator.error_frames("boom", code="upstream_error")
    events = _parse_frames(frames)
    assert events[0]["type"] == "error"
    assert events[1]["type"] == "response.failed"
    assert events[1]["response"]["status"] == "failed"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed")
