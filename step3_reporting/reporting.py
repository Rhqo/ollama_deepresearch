from typing import List
from pydantic import BaseModel, Field
from ollama import chat

class ReportResponse(BaseModel):
    report: str = Field(..., description="생성된 마크다운 형식 보고서")

def write_final_report(
    prompt: str,
    learnings: List[str],
    visited_urls: List[str],
    client,
    model: str,
) -> str:
    """Ollama를 사용한 구조화된 보고서 생성"""
    
    # 학습 내용 포매팅
    learnings_formatted = "\n".join(
        [f"### 학습 내용 {i+1}\n{learning}" 
         for i, learning in enumerate(learnings)]
    )[:150000]

    # 시스템 프롬프트 구성
    system_msg = """## 보고서 작성 지침
1. 마크다운 형식 준수
2. 6000자 이상의 상세한 내용
3. 학술 논문 수준의 체계적 구조
4. 데이터 기반 결론 포함
5. 표 및 리스트 적극 활용"""

    # 사용자 프롬프트 구성
    user_prompt = f"""
# 연구 주제
{prompt}

# 주요 학습 내용
{learnings_formatted}

## 보고서 요구사항
- 서론: 연구 배경 및 목적 설명
- 본론: 학습 내용을 주제별로 분류하여 심층 분석
- 결론: 종합적 요약 및 향후 연구 방향 제시
- 부록: 참고 문헌 목록 포함
"""

    try:
        # Ollama API 호출
        response = client(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt}
            ],
            format=ReportResponse.model_json_schema(),
            options={'temperature': 0.5}
        )
        
        # 응답 파싱
        parsed = ReportResponse.model_validate_json(response.message.content)
        report_body = parsed.report
        
        # 출처 추가
        sources_section = "\n\n## 참고 문헌\n" + "\n".join(
            [f"- [{url}]({url})" for url in visited_urls]
        )
        
        return report_body + sources_section

    except Exception as e:
        print(f"보고서 생성 오류: {e}")
        print(f"원본 응답: {response.message.content if hasattr(response, 'message') else '없음'}")
        return "# 보고서 생성 실패\n\n문제가 지속되면 모델을 변경해 보세요."

# 도우미 함수 (필요시 추가)
def format_markdown_section(title: str, content: str) -> str:
    return f"\n\n## {title}\n\n{content}"
