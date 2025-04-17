from typing import List
from pydantic import BaseModel, Field
from ollama import chat

class FeedbackResponse(BaseModel):
    questions: List[str] = Field(..., description="생성된 후속 질문 목록")

def generate_feedback(query: str, client, model: str, max_feedbacks: int = 3) -> List[str]:
    """연구 방향을 명확히 하기 위한 후속 질문 생성 (Ollama 버전)"""
    
    system_msg = """당신은 연구 주제 분석 전문가입니다. 다음 규칙을 따라주세요:
1. 사용자의 초기 질문을 더 구체화할 수 있는 후속 질문 생성
2. 최대 3개의 질문만 생성
3. 반드시 한국어로 작성
4. JSON 형식으로 응답"""

    user_prompt = f"""
# 연구 주제 분석 요청
{query}

## 출력 요구사항
- questions 배열에 질문을 포함
- 각 질문은 1~2문장으로 구성
- 전문 용어 사용 권장
- 구체적인 사례 요청 질문 포함
"""

    try:
        # Ollama API 호출
        response = chat(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt}
            ],
            format=FeedbackResponse.model_json_schema(),
            options={'temperature': 0.7}
        )
        
        # 응답 파싱
        parsed = FeedbackResponse.model_validate_json(response.message.content)
        return parsed.questions[:max_feedbacks]
    
    except Exception as e:
        print(f"후속 질문 생성 오류: {str(e)}")
        print(f"원본 응답: {response.message.content if hasattr(response, 'message') else '없음'}")
        return []
