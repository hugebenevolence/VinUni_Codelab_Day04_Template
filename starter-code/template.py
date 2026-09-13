"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import os
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
Bạn là VinAssistant — trợ lý AI chính thức của hệ sinh thái Vingroup.

## PERSONA
- Tên: VinAssistant
- Vai trò: Chuyên viên tư vấn sản phẩm & dịch vụ VinFast, Vinpearl và tiếp nhận hỗ trợ khách hàng.
- Giọng nói: Chuyên nghiệp, thân thiện, chính xác, luôn ưu tiên sự thật hơn là suy đoán.

## AVAILABLE TOOLS
1. search_product_catalog(category, max_price) — Tra cứu sản phẩm/dịch vụ Vingroup theo danh mục ('xe_dien' hoặc 'du_lich') và giá tối đa (VNĐ).
2. submit_support_ticket(customer_name, issue_description, priority) — Ghi nhận yêu cầu hỗ trợ của khách hàng vào hệ thống ticket.

## CORE RULES
1. KHÔNG BAO GIỜ bịa dữ liệu sản phẩm, giá cả hay chính sách. PHẢI gọi tool để lấy dữ liệu thực khi câu hỏi liên quan tới sản phẩm/dịch vụ cụ thể.
2. Nếu khách hàng báo lỗi, sự cố hoặc cần hỗ trợ, PHẢI gọi submit_support_ticket để ghi nhận, không được chỉ trả lời suông.
3. Nếu không tìm thấy kết quả phù hợp, thông báo rõ ràng cho khách hàng thay vì im lặng hoặc bịa ra sản phẩm.

## OPERATIONAL BOUNDARIES
- Chỉ trả lời các câu hỏi liên quan tới sản phẩm/dịch vụ của hệ sinh thái Vingroup (VinFast, Vinpearl, hỗ trợ khách hàng).
- Từ chối lịch sự với các yêu cầu nằm ngoài phạm vi trên.

## OUTPUT CONTRACT
Mỗi lượt xử lý tuân theo định dạng:
Thought: <suy luận về intent và tool cần dùng>
Action: <tên tool được gọi, nếu có>
Observation: <kết quả trả về từ tool>
Final Answer: <câu trả lời cuối cùng, ngắn gọn, chính xác, dựa trên Observation>
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        # TODO 2: Trả về câu trả lời tĩnh (mock) hoặc gọi OpenAI API 1 lượt (không dùng tool)
        # Mục tiêu: Quan sát hiện tượng bịa thông tin (hallucination)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return {
                "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
                "tool_calls": [],
                "status": "success",
                "mode": "mock_baseline"
            }

        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Bạn là trợ lý AI của Vingroup, trả lời câu hỏi khách hàng về sản phẩm VinFast, Vinpearl."
                    },
                    {"role": "user", "content": user_input}
                ]
            )
            answer = response.choices[0].message.content
            return {
                "answer": answer,
                "tool_calls": [],
                "status": "success",
                "mode": "live_openai"
            }
        except Exception as e:
            return {
                "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
                "tool_calls": [],
                "status": "success",
                "mode": "mock_baseline_fallback",
                "error": str(e)
            }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    @staticmethod
    def _detect_category(text_lower: str) -> str:
        if "xe điện" in text_lower or "vinfast" in text_lower:
            return "xe_dien"
        if "du lịch" in text_lower or "vinpearl" in text_lower or "nghỉ dưỡng" in text_lower:
            return "du_lich"
        return None

    @staticmethod
    def _extract_customer_name(user_input: str) -> str:
        match = re.search(r"tôi tên ([^,.]+)", user_input, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return "Khách hàng"

    @staticmethod
    def _extract_priority(text_lower: str) -> str:
        if any(kw in text_lower for kw in ["gấp", "nghiêm trọng", "khẩn cấp"]):
            return "high"
        if any(kw in text_lower for kw in ["không gấp", "bình thường"]):
            return "low"
        return "medium"

    @staticmethod
    def _faq_answer(text_lower: str) -> str:
        if "bảo hành" in text_lower:
            return "Chính sách bảo hành pin xe điện VinFast kéo dài 10 năm hoặc 160.000 km, tùy điều kiện nào đến trước."
        return "Xin lỗi, tôi chưa có thông tin cụ thể cho câu hỏi này. Vui lòng liên hệ tổng đài Vingroup để được hỗ trợ thêm."

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []
        text_lower = user_input.lower()

        # TODO 3: Phân tích intent từ user_input
        category = self._detect_category(text_lower)
        price_match = re.search(r"(\d+)\s*triệu", text_lower)
        max_price = int(price_match.group(1)) * 1_000_000 if price_match else 999999999999

        search_intent_keywords = ["xem", "tìm", "muốn mua", "có sản phẩm", "giá", "còn", "nào"]
        needs_catalog = category is not None and (
            price_match is not None or any(kw in text_lower for kw in search_intent_keywords)
        )

        ticket_keywords = ["lỗi", "hỏng", "sự cố", "hỗ trợ", "khiếu nại", "vấn đề"]
        needs_ticket = any(kw in text_lower for kw in ticket_keywords)

        is_faq = not needs_catalog and not needs_ticket

        intents = {"needs_catalog": needs_catalog, "needs_ticket": needs_ticket, "is_faq": is_faq}
        self.trace.append({"step": "intent_detection", "intents": intents})

        # TODO 4: Xây dựng Agent Loop
        iteration = 1
        answer_parts: List[str] = []

        while iteration <= self.max_iterations:
            if needs_catalog:
                results = search_product_catalog(category=category, max_price=max_price)
                self.trace.append({
                    "step": f"iteration_{iteration}",
                    "action": "search_product_catalog",
                    "action_input": {"category": category, "max_price": max_price},
                    "observation": results
                })
                if not results or len(results) == 0:
                    answer_parts.append("Rất tiếc, không tìm thấy sản phẩm phù hợp.")
                else:
                    names = ", ".join(p["name"] for p in results)
                    answer_parts.append(f"Chúng tôi tìm thấy các sản phẩm phù hợp: {names}.")
                needs_catalog = False
                iteration += 1
                if not needs_ticket:
                    break
                continue

            if needs_ticket:
                customer_name = self._extract_customer_name(user_input)
                priority = self._extract_priority(text_lower)
                ticket = submit_support_ticket(
                    customer_name=customer_name,
                    issue_description=user_input,
                    priority=priority
                )
                self.trace.append({
                    "step": f"iteration_{iteration}",
                    "action": "submit_support_ticket",
                    "action_input": {"customer_name": customer_name, "priority": priority},
                    "observation": ticket
                })
                answer_parts.append(
                    f"Đã ghi nhận yêu cầu hỗ trợ của {customer_name}, mã ticket {ticket['ticket_id']}."
                )
                needs_ticket = False
                iteration += 1
                break

            if is_faq:
                answer_parts.append(self._faq_answer(text_lower))
                self.trace.append({
                    "step": f"iteration_{iteration}",
                    "action": "faq_static_answer",
                    "observation": "no_tool_call"
                })
                iteration += 1
                break

            break

        final_answer = " ".join(answer_parts) if answer_parts else "Không có yêu cầu nào cần xử lý."
        return {
            "answer": final_answer,
            "trace": self.trace,
            "iterations": iteration - 1,
            "status": "completed"
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
