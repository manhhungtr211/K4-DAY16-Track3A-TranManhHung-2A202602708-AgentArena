"""Kiểm thử chuyên sâu cho CitationChecker và Critic:
1. Unsupported claim: Khẳng định thông tin nhưng không có nguồn hoặc nguồn rỗng.
2. Misattributed claim: Trích dẫn sai tài liệu (câu có thật nhưng gắn sai doc_id).
3. Fabricated source: Bịa ra doc_id không hề tồn tại trong kho tài liệu.
4. Loop detection: Kiểm tra Agent có bị lặp lại hành động/câu trả lời qua nhiều lượt hay không.
"""

from __future__ import annotations

import pytest
from arena.corpus import Corpus
from arena.model import MockModel
from arena.tools import Tools
from arena.trace import Trace
from harness.agent import ReActAgent
from harness.layers.citation_checker import CitationChecker
from harness.layers.critic import Critic


SEED = 42
CORPUS = Corpus.generate(seed=SEED)

# Lấy 2 tài liệu mẫu từ Corpus
DOC_A = CORPUS.docs[0]
DOC_B = CORPUS.docs[1]

# Lấy một dòng câu văn có thật từ DOC_A
LINE_A = next(line.strip() for line in DOC_A.body.splitlines() if len(line.strip()) > 20)


class MockContext:
    """Mock context giả lập môi trường AgentContext để kiểm thử nhanh middleware."""

    def __init__(self, corpus: Corpus, observed_text: str):
        self.corpus = corpus
        self.observed_text = observed_text

    def saw(self, text: str) -> bool:
        return bool(text) and text in self.observed_text


# ==============================================================================
# 1. Unsupported Claim: Khẳng định thông tin nhưng không có nguồn / bịa đặt
# ==============================================================================
def test_unsupported_claim_is_eliminated_and_triggers_abstain():
    """Khẳng định một câu hoàn toàn không có trong bất kỳ tài liệu quan sát nào:
    Critic phải phát hiện và xoá claim này, đồng thời bật abstain=True."""
    checker = CitationChecker()
    critic = Critic()

    ctx = MockContext(corpus=CORPUS, observed_text=DOC_A.body)

    report = {
        "answer": "Doanh thu năm nay tăng 200%.",
        "claims": [
            {
                "text": "Doanh thu công ty tăng trưởng vượt bậc 200% trong năm nay.",
                "doc_id": DOC_A.doc_id,  # Gắn doc_A nhưng doc_A không hề nói câu này
            }
        ],
        "citations": [DOC_A.doc_id],
        "abstain": False,
    }

    # Chạy lần lượt qua citation_checker rồi đến critic (như thứ tự after_agent)
    report = checker.after_agent(ctx, report)
    report = critic.after_agent(ctx, report)

    # Kết quả: claim bịa đặt không có căn cứ phải bị xoá sạch và bật abstain
    assert len(report["claims"]) == 0, "Claim không có căn cứ phải bị Critic loại bỏ"
    assert report["abstain"] is True, "Phải bật abstain=True khi không có bằng chứng hỗ trợ"
    assert report["citations"] == [], "Citations phải được dọn sạch về rỗng"


# ==============================================================================
# 2. Misattributed Claim: Dẫn sai tài liệu (câu có thật nhưng gán nhầm doc_id khác)
# ==============================================================================
def test_misattributed_claim_is_corrected_to_real_source():
    """Câu văn nằm trong DOC_A, nhưng model lại gắn nhầm vào DOC_B:
    CitationChecker phải tự động phát hiện và đổi lại đúng doc_id của DOC_A."""
    checker = CitationChecker()

    # Agent đã quan sát cả DOC_A và DOC_B
    ctx = MockContext(corpus=CORPUS, observed_text=DOC_A.body + "\n" + DOC_B.body)

    report = {
        "answer": "Thông tin SLA nội bộ.",
        "claims": [
            {
                "text": LINE_A,
                "doc_id": DOC_B.doc_id,  # GÁN SAI: câu của DOC_A nhưng gán nhầm vào DOC_B
            }
        ],
        "citations": [DOC_B.doc_id],
        "abstain": False,
    }

    report = checker.after_agent(ctx, report)

    # Kiểm tra doc_id đã được sửa lại đúng DOC_A
    assert report["claims"][0]["doc_id"] == DOC_A.doc_id, (
        f"Misattribution! Phải đổi doc_id sang {DOC_A.doc_id}, nhưng vẫn là {report['claims'][0]['doc_id']}"
    )
    # Tuyệt đối giữ nguyên câu chữ (Provenance)
    assert report["claims"][0]["text"] == LINE_A, "Câu chữ của claim không được phép bị thay đổi!"
    assert DOC_A.doc_id in report["citations"], "Citations phải chứa tài liệu đúng"


# ==============================================================================
# 3. Fabricated Source: Bịa ra doc_id không hề tồn tại trong hệ thống
# ==============================================================================
def test_fabricated_source_is_rescued_or_removed():
    """Trường hợp 3a: Câu văn có thật nhưng model bịa ra doc_id không tồn tại ('doc-9999').
    CitationChecker phải gán lại về tài liệu thật."""
    checker = CitationChecker()
    critic = Critic()

    ctx = MockContext(corpus=CORPUS, observed_text=DOC_A.body)

    # 3a. Câu có thật, nguồn bịa
    report_rescued = {
        "answer": "Báo cáo nội bộ.",
        "claims": [{"text": LINE_A, "doc_id": "doc-9999-fake"}],
        "citations": ["doc-9999-fake"],
        "abstain": False,
    }
    report_rescued = checker.after_agent(ctx, report_rescued)
    assert report_rescued["claims"][0]["doc_id"] == DOC_A.doc_id, (
        "CitationChecker phải cứu claim có thật bằng cách trỏ về doc_id có thật trong quan sát"
    )

    # 3b. Cả câu lẫn nguồn đều bịa hoàn toàn
    report_fake = {
        "answer": "Thông tin bịa đặt.",
        "claims": [{"text": "Hệ thống kho mở cửa 25 tiếng mỗi ngày.", "doc_id": "doc-9999-fake"}],
        "citations": ["doc-9999-fake"],
        "abstain": False,
    }
    report_fake = checker.after_agent(ctx, report_fake)
    report_fake = critic.after_agent(ctx, report_fake)

    assert len(report_fake["claims"]) == 0, "Nguồn bịa và câu bịa phải bị xoá bỏ hoàn toàn"
    assert "doc-9999-fake" not in report_fake["citations"], "doc_id bịa không được phép có trong citations"


# ==============================================================================
# 4. Loop Detection: Kiểm tra Agent có bị lặp hành động / câu trả lời sai không
# ==============================================================================
def test_detect_agent_action_looping_behavior():
    """Kiểm chứng hành vi: Agent baseline CÓ BỊ LẶP hành động/câu trả lời sai không?
    Thực tế: Mô hình ReAct baseline cố tình mắc lỗi lặp lại các lượt gọi tool vô nghĩa
    hoặc gọi lại cùng một tài liệu nhiều lần khi chưa có layer budget/retry."""
    trace = Trace(run_id="test-loop-check", seed=SEED)
    tools = Tools(corpus=CORPUS, trace=trace, seed=SEED)
    model = MockModel(corpus=CORPUS, seed=SEED)

    agent = ReActAgent(
        model=model,
        tools=tools,
        trace=trace,
        middleware=[CitationChecker(), Critic()],
        corpus=CORPUS,
    )

    brief = {
        "brief_id": "test-loop",
        "question_vi": "Thời gian giao hàng cam kết SLA nội thành hiện nay là bao nhiêu ngày?",
        "budget": {"max_tool_calls": 8, "max_tokens": 12000, "max_seconds": 60},
    }

    report = agent.run(brief)

    # Lấy danh sách các tool calls đã được ghi lại trong trace
    events = trace._events
    tool_calls = [e for e in events if e.get("event") == "tool_call"]

    # Đếm số lần Agent gọi các hành động trùng lặp liên tiếp
    repeated_loops = []
    for i in range(len(tool_calls) - 1):
        curr_call = (tool_calls[i].get("name"), str(tool_calls[i].get("args")))
        next_call = (tool_calls[i + 1].get("name"), str(tool_calls[i + 1].get("args")))
        if curr_call == next_call:
            repeated_loops.append((i, curr_call))

    # Báo cáo phát hiện hiện tượng lặp (Đây là điểm yếu có chủ đích của baseline agent)
    has_loop = len(repeated_loops) > 0
    print(f"\n[PHÁT HIỆN LOOP]: Agent có {len(repeated_loops)} lần gọi tool lặp lại liên tiếp!")
    for step, call in repeated_loops:
        print(f"  - Bước {step}: lặp lại lệnh {call}")

    # Khẳng định: Hệ thống phát hiện được hành vi lặp của agent
    assert has_loop is True, "Agent ReAct baseline phải bộc lộ điểm yếu lặp lại công cụ (để cần tới budget_policy và retry)!"
    assert isinstance(report, dict) and "answer" in report

