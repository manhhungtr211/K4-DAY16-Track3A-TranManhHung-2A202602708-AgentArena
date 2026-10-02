"""Script soi chi tiết Report trước và sau khi đi qua Critic layer.
Chạy: python scripts/inspect_critic.py
Xoá: Bạn có thể xoá file này bất cứ lúc nào.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from arena.briefs import load_public_briefs
from arena.corpus import Corpus
from arena.model import MockModel
from arena.tools import Tools
from arena.trace import Trace
from harness.agent import ReActAgent
from harness.layers.critic import Critic


def inspect():
    briefs = load_public_briefs()
    corpus = Corpus.generate(seed=42)

    print("=" * 80)
    print("[SOI CHI TIET KET QUA TRUOC VA SAU KHI QUA CRITIC LAYER]")
    print("=" * 80)

    for b in briefs:
        brief_id = b["brief_id"] if isinstance(b, dict) else b.brief_id
        question = b.get("question_vi", b.get("question", "")) if isinstance(b, dict) else getattr(b, "question_vi", getattr(b, "question", ""))
        title = b.get("title", "") if isinstance(b, dict) else getattr(b, "title", "")
        budget = b.get("budget") if isinstance(b, dict) else getattr(b, "budget", None)

        # 1. Chạy Agent KHÔNG CÓ layer nào (để lấy kết quả thô của Model)
        raw_trace = Trace(run_id=f"raw-{brief_id}", seed=42)
        raw_tools = Tools(corpus=corpus, trace=raw_trace, seed=42)
        raw_model = MockModel(corpus=corpus, seed=42)
        raw_agent = ReActAgent(
            model=raw_model,
            tools=raw_tools,
            trace=raw_trace,
            middleware=[],
            corpus=corpus,
        )
        raw_report = raw_agent.run(b)

        # 2. Chạy Agent CÓ Critic layer
        crit_trace = Trace(run_id=f"crit-{brief_id}", seed=42)
        crit_tools = Tools(corpus=corpus, trace=crit_trace, seed=42)
        crit_model = MockModel(corpus=corpus, seed=42)
        crit_agent = ReActAgent(
            model=crit_model,
            tools=crit_tools,
            trace=crit_trace,
            middleware=[Critic()],
            corpus=corpus,
        )
        crit_report = crit_agent.run(b)



        # 3. Hiển thị so sánh
        print(f"\n📋 BRIEF: [{brief_id}] - {title}")
        print(f"❓ Câu hỏi: {question}")

        raw_claims = raw_report.get("claims", [])
        crit_claims = crit_report.get("claims", [])

        print(f"  🔹 [TRƯỚC CRITIC] : {len(raw_claims)} claims | Abstain: {raw_report.get('abstain')}")
        for idx, c in enumerate(raw_claims, 1):
            print(f"     {idx}. doc_id: {c.get('doc_id')} | «{c.get('text', '')[:70]}...»")

        print(f"  🔸 [SAU CRITIC]   : {len(crit_claims)} claims | Abstain: {crit_report.get('abstain')}")
        if crit_claims:
            for idx, c in enumerate(crit_claims, 1):
                print(f"     {idx}. doc_id: {c.get('doc_id')} | «{c.get('text', '')[:70]}...»")
        else:
            print("     (Đã loại bỏ hết claim do không có căn cứ)")

        # Phân tích chênh lệch
        if len(raw_claims) > len(crit_claims):
            print(f"  ❌ ĐÃ XOÁ BỎ {len(raw_claims) - len(crit_claims)} claim bịa đặt / mâu thuẫn.")
        elif raw_report.get("abstain") != crit_report.get("abstain"):
            print(f"  ⚠️ ĐÃ ĐỔI TRẠNG THÁI ABSTAIN: {raw_report.get('abstain')} ➔ {crit_report.get('abstain')}")
        else:
            print("  ✅ Giữ nguyên toàn bộ claims hợp lệ.")
        print("-" * 80)


if __name__ == "__main__":
    inspect()
