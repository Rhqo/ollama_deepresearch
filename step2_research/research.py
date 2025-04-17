import os
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from ollama import chat
from firecrawl import FirecrawlApp

# 데이터 모델 정의
class SearchResult(BaseModel):
    url: str
    markdown: str
    description: str
    title: str

class SerpQuery(BaseModel):
    query: str
    research_goal: str = Field(..., description="검색 목적 설명")

class SerpQueryResponse(BaseModel):
    queries: List[SerpQuery] = Field(..., description="생성된 검색 쿼리 목록")

class ResearchResult(BaseModel):
    learnings: List[str] = Field(..., description="연구에서 얻은 주요 결과")
    visited_urls: List[str] = Field(..., description="방문한 URL 목록")

class SerpResultResponse(BaseModel):
    learnings: List[str] = Field(..., description="추출된 학습 내용")
    followUpQuestions: List[str] = Field(..., description="후속 연구 질문")

def firecrawl_search(query: str, timeout: int = 15000, limit: int = 5) -> List[SearchResult]:
    """Firecrawl API를 사용한 동기 검색"""
    try:
        app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY", ""))
        response = app.search(
            query=query,
            params={
                "timeout": timeout,
                "limit": limit,
                "scrapeOptions": {"formats": ["markdown"]}
            }
        )
        valid_results = []
        for item in response.get("data", []):
            if "markdown" in item and item["markdown"]:
                valid_results.append(SearchResult(**item))
        return valid_results
    except Exception as e:
        print(f"Firecrawl 검색 오류: {e}")
        return []

def generate_serp_queries(
    query: str,
    client,
    model: str,
    num_queries: int = 3,
    learnings: Optional[List[str]] = None,
) -> List[SerpQuery]:
    """Ollama 기반 구조화된 검색 쿼리 생성"""
    system_prompt = "연구 목적에 맞는 검색 쿼리를 생성하는 AI입니다."
    user_prompt = f"""
    [주요 연구 주제]
    {query}
    
    [이전 연구 결과]
    {learnings if learnings else '없음'}
    
    위 내용을 바탕으로 구체적인 검색 쿼리를 {num_queries}개 생성해주세요.
    각 쿼리에는 'query'와 'research_goal' 필드가 포함되어야 합니다.
    """

    try:
        response = client(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            format=SerpQueryResponse.model_json_schema()
        )
        
        parsed = SerpQueryResponse.model_validate_json(response.message.content)
        return parsed.queries[:num_queries]
    except Exception as e:
        print(f"검색 쿼리 생성 오류: {e}")
        return []

def process_serp_result(
    query: str,
    search_result: List[SearchResult],
    client,
    model: str,
    num_learnings: int = 5,
    num_follow_up_questions: int = 3,
) -> Dict[str, List[str]]:
    """검색 결과 분석 및 구조화"""
    contents = [item.markdown[:25000] for item in search_result if item.markdown]
    contents_str = "\n".join([f"<문서>\n{content}\n</문서>" for content in contents])
    
    system_prompt = "검색 결과를 분석해 핵심 내용을 추출하는 AI입니다."
    user_prompt = f"""
    [분석 요청]
    검색 쿼리: {query}
    문서 개수: {len(contents)}
    
    다음 내용에서:
    1. 주요 학습 내용 {num_learnings}개 추출
    2. 후속 연구 질문 {num_follow_up_questions}개 생성
    """
    
    try:
        response = client(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "user", "content": contents_str}
            ],
            format=SerpResultResponse.model_json_schema()
        )
        
        parsed = SerpResultResponse.model_validate_json(response.message.content)
        return {
            "learnings": parsed.learnings[:num_learnings],
            "followUpQuestions": parsed.followUpQuestions[:num_follow_up_questions]
        }
    except Exception as e:
        print(f"결과 분석 오류: {e}")
        return {"learnings": [], "followUpQuestions": []}

def deep_research(
    query: str,
    breadth: int,
    depth: int,
    client,
    model: str,
    learnings: Optional[List[str]] = None,
    visited_urls: Optional[List[str]] = None,
) -> ResearchResult:
    """심층 연구 수행 재귀 함수"""
    current_learnings = learnings.copy() if learnings else []
    current_urls = visited_urls.copy() if visited_urls else []
    
    print(f"\n{'='*30} 연구 시작 (깊이 {depth}) {'='*30}")
    print(f"주제: {query}\n")

    # 1단계: 검색 쿼리 생성
    serp_queries = generate_serp_queries(
        query=query,
        client=client,
        model=model,
        num_queries=breadth,
        learnings=current_learnings
    )
    
    # 2단계: 각 쿼리 실행
    for idx, serp_query in enumerate(serp_queries, 1):
        print(f"[{idx}/{len(serp_queries)}] 검색 실행: {serp_query.query}")
        
        # 검색 수행
        results = firecrawl_search(serp_query.query)
        new_urls = [result.url for result in results]
        print(f"→ 발견된 URL: {len(new_urls)}개")
        
        # 결과 처리
        processed = process_serp_result(
            query=serp_query.query,
            search_result=results,
            client=client,
            model=model
        )
        
        # 데이터 누적
        current_learnings += processed["learnings"]
        current_urls += new_urls
        
        # 3단계: 재귀 호출
        if depth > 1:
            print(f"\n{'→'*3} 후속 연구 시작 (깊이 {depth-1})")
            sub_result = deep_research(
                query=processed["followUpQuestions"][0],
                breadth=max(1, breadth//2),
                depth=depth-1,
                client=client,
                model=model,
                learnings=current_learnings,
                visited_urls=current_urls
            )
            current_learnings = sub_result.learnings
            current_urls = sub_result.visited_urls

    return ResearchResult(
        learnings=list(set(current_learnings)),
        visited_urls=list(set(current_urls))
    )
