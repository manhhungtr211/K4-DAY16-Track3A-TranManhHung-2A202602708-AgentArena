"""Kiểm thử hợp đồng kỹ thuật cho hook `after_agent` trong Middleware."""

from __future__ import annotations

import pytest
from arena.corpus import Corpus
from arena.model import MockModel
from arena.tools import Tools
from arena.trace import Trace
from harness.agent import ReActAgent
from harness.middleware import Middleware


SEED = 42
CORPUS = Corpus.generate(seed=SEED)
QUESTION = "Thời gian giao hàng cam kết SLA nội thành hiện nay là bao nhiêu ngày?"
BRIEF = {
    "brief_id": "test-sla",
    "question_vi": QUESTION,
    "budget": {"max_tool_calls": 8, "max_tokens": 12000, "max_seconds": 60},
}


def _make_agent(middleware_list):
    """Helper khởi tạo agent nhanh cho test."""
    trace = Trace(run_id="test-run", seed=SEED)
    tools = Tools(corpus=CORPUS, trace=trace, seed=SEED)
    model = MockModel(corpus=CORPUS, seed=SEED)
    return ReActAgent(
        model=model,
        tools=tools,
        trace=trace,
        middleware=middleware_list,
        corpus=CORPUS,
    )


# ==============================================================================
# 1. test_runs_once_per_invocation: Chạy ĐÚNG 1 LẦN duy nhất
# ==============================================================================
def test_runs_once_per_invocation():
    """Đảm bảo `after_agent` chỉ được gọi ĐÚNG 1 LẦN khi kết thúc toàn bộ luồng,
    không bị lặp bên trong vòng lặp ReAct."""
    call_count = 0

    class CounterMiddleware(Middleware):
        name = "counter"

        def after_agent(self, ctx, report):
            nonlocal call_count
            call_count += 1
            return report

    agent = _make_agent([CounterMiddleware()])
    report = agent.run(BRIEF)

    assert isinstance(report, dict), "Agent phải trả về một report dict hợp lệ"
    assert call_count == 1, f"`after_agent` phải chạy đúng 1 lần, nhưng đã chạy {call_count} lần!"


# ==============================================================================
# 2. test_runs_when_agent_stops_early: Luôn chạy dù dừng sớm (Guaranteed Cleanup)
# ==============================================================================
def test_runs_when_agent_stops_early():
    """Dù Agent bị ngắt sớm (ví dụ do max_steps=1 hoặc budget cạn),
    `after_agent` VẪN BẮT BUỘC PHẢI CHẠY (như khối finally)."""
    ran_cleanup = False

    class CleanupMiddleware(Middleware):
        name = "cleanup"

        def after_agent(self, ctx, report):
            nonlocal ran_cleanup
            ran_cleanup = True
            return report

    # Ép agent dừng ngay sau 1 bước (max_steps=1)
    agent = _make_agent([CleanupMiddleware()])
    agent.max_steps = 1
    report = agent.run(BRIEF)

    assert ran_cleanup is True, "`after_agent` phải luôn được gọi ngay cả khi agent dừng sớm!"


# ==============================================================================
# 3. test_save_failure_does_not_lose_answer: Dung lỗi (Fault Tolerance)
# ==============================================================================
def test_save_failure_does_not_lose_answer():
    """Nếu logic phụ trong `after_agent` gặp lỗi lưu log/metrics,
    middleware được thiết kế an toàn không làm mất `answer` cốt lõi của model."""

    class SafeLoggingMiddleware(Middleware):
        name = "safe_logging"

        def after_agent(self, ctx, report):
            # Thử lưu metrics/log nhưng giả lập bị lỗi (ví dụ disk full/network drop)
            try:
                raise IOError("Mất kết nối tới máy chủ log")
            except Exception:
                pass  # Nuốt lỗi phụ để bảo vệ kết quả chính
            return report

    agent = _make_agent([SafeLoggingMiddleware()])
    report = agent.run(BRIEF)


    assert "answer" in report, "Report không được mất trường 'answer'"
    assert len(report["answer"]) > 0, "Đáp án chính phải còn nguyên vẹn dù log gặp sự cố"


# ==============================================================================
# 4. test_save_is_idempotent: Tính đẳng thế (Idempotency)
# ==============================================================================
def test_save_is_idempotent():
    """Gọi `after_agent` nhiều lần liên tiếp trên cùng 1 report không làm
    thay đổi trạng thái hay tạo dữ liệu trùng lặp (duplicate citations/claims)."""

    class DeduplicateMiddleware(Middleware):
        name = "dedup"

        def after_agent(self, ctx, report):
            # Sắp xếp và loại bỏ trùng lặp citation
            citations = report.get("citations", [])
            report["citations"] = sorted(list(set(citations)))
            return report

    mw = DeduplicateMiddleware()
    initial_report = {
        "answer": "SLA giao hàng là 2 ngày.",
        "claims": [{"text": "SLA là 2 ngày", "doc_id": "doc-0001"}],
        "citations": ["doc-0001", "doc-0001"],  # Bị trùng citation
        "abstain": False,
    }

    # Lần gọi 1
    res1 = mw.after_agent(None, dict(initial_report))
    # Lần gọi 2 (trên kết quả của lần 1)
    res2 = mw.after_agent(None, dict(res1))

    assert res1 == res2, "Kết quả của lần chạy thứ 2 phải đồng nhất tuyệt đối với lần 1 (Idempotent)"
    assert len(res2["citations"]) == 1, "Citations không được nhân bản"
