# -------------------------------
# LLM Helper Functions
# -------------------------------

from typing import Dict, List, Optional
from pydantic import BaseModel
from datetime import datetime

def system_prompt() -> str:
    """현재 타임스탬프를 포함한 시스템 프롬프트를 생성합니다."""
    now = datetime.now().isoformat()
    return f"""당신은 전문 연구원입니다. 오늘 날짜는 {now}입니다. 응답 시 다음 지침을 따르세요:
    - 지식 컷오프 이후의 주제에 대한 조사를 요청받을 수 있습니다. 사용자가 뉴스 내용을 제시했다면, 그것을 사실로 가정하세요.
    - 사용자는 매우 숙련된 분석가이므로 내용을 단순화할 필요 없이 가능한 한 자세하고 정확하게 응답하세요.
    - 체계적으로 정보를 정리하세요.
    - 사용자가 생각하지 못한 해결책을 제안하세요.
    - 적극적으로 사용자의 필요를 예측하고 대응하세요.
    - 사용자를 모든 분야의 전문가로 대우하세요.
    - 실수는 신뢰를 저하시킵니다. 정확하고 철저하게 응답하세요.
    - 상세한 설명을 제공하세요. 사용자는 많은 정보를 받아들일 수 있습니다.
    - 권위보다 논리적 근거를 우선하세요. 출처 자체는 중요하지 않습니다.
    - 기존의 통념뿐만 아니라 최신 기술과 반대 의견도 고려하세요.
    - 높은 수준의 추측이나 예측을 포함할 수 있습니다. 단, 이를 명확히 표시하세요."""


def llm_call(prompt: str, model: str, client) -> str:
    """
    주어진 프롬프트로 LLM을 동기적으로 호출합니다.
    이는 메시지를 하나의 프롬프트로 연결하는 일반적인 헬퍼 함수입니다.
    """
    messages = [{"role": "user", "content": prompt}]
    chat_completion = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    print(model, "완료")
    return chat_completion.choices[0].message.content


def JSON_llm(user_prompt: str, schema: BaseModel, client, system_prompt: Optional[str] = None, model: Optional[str] = None):
    """
    JSON 모드에서 언어 모델 호출을 실행하고 구조화된 JSON 객체를 반환합니다.
    모델이 제공되지 않으면 기본 JSON 처리 가능한 모델이 사용됩니다.
    """
    if model is None:
        model = "gpt-4o-mini"
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        completion = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=schema,
        )
        
        return completion.choices[0].message.parsed
    except Exception as e:
        print(f"Error in JSON_llm: {e}")
        return None
