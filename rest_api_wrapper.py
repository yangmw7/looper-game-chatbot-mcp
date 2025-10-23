from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import logging
from typing import List, Dict, Any, Optional
import uvicorn
import openai
import os

# 기존 MCP 서버 모듈들 import
from server import MariaDBServer

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MCP REST API Wrapper", version="1.0.0")

# 전역 MCP 서버 인스턴스
mcp_server: Optional[MariaDBServer] = None

# OpenAI 클라이언트 초기화
openai_client = openai.AsyncOpenAI()

class SearchRequest(BaseModel):
    user_query: str
    database_name: str = "mcp_db"
    vector_store_name: str = "knowledge_base"
    k: int = 5
    generate_answer: bool = True  # 자연어 답변 생성 여부

class SearchResponse(BaseModel):
    success: bool
    results: List[Dict[str, Any]]
    answer: Optional[str] = None  # 생성된 자연어 답변
    error: Optional[str] = None

async def generate_answer(query: str, search_results: List[Dict]) -> str:
    """검색 결과를 바탕으로 자연어 답변 생성"""
    try:
        if not search_results:
            return "죄송합니다. 관련 정보를 찾을 수 없습니다."
        
        # 검색 결과에서 문서 내용 추출
        context = "\n".join([result["document"] for result in search_results])
        
        prompt = f"""사용자 질문: {query}

관련 정보:
{context}

위 정보를 바탕으로 사용자의 질문에 대해 한국어로 자연스럽고 도움이 되는 답변을 해주세요. 
답변은 친근하고 이해하기 쉽게 작성해주세요."""
        
        # OpenAI API 호출
        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"답변 생성 중 오류 발생: {e}")
        # OpenAI API 오류 시 기본 답변 반환
        if search_results:
            return f"다음 정보를 찾았습니다: {search_results[0]['document']}"
        return "답변 생성 중 오류가 발생했습니다."
@app.on_event("startup")
async def startup_event():
    """서버 시작 시 MCP 서버 초기화"""
    global mcp_server
    try:
        logger.info("MCP 서버 생성 시작...")
        mcp_server = MariaDBServer()
        
        logger.info("데이터베이스 풀 초기화 시작...")
        await mcp_server.initialize_pool()
        
        logger.info("도구 등록 시작...")
        mcp_server.register_tools()
        
        logger.info("MCP 서버 초기화 완료")
    except Exception as e:
        logger.error(f"MCP 서버 초기화 실패: {e}")
        logger.error(f"오류 상세: {str(e)}")
        import traceback
        logger.error(f"스택 트레이스: {traceback.format_exc()}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 정리"""
    global mcp_server
    if mcp_server:
        await mcp_server.close_pool()
        logger.info("MCP 서버 종료 완료")

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {"status": "healthy", "message": "MCP REST API is running"}

@app.post("/search", response_model=SearchResponse)
async def search_vector_store(request: SearchRequest):
    """벡터 스토어 검색 및 자연어 답변 생성 엔드포인트"""
    if not mcp_server:
        raise HTTPException(status_code=500, detail="MCP 서버가 초기화되지 않았습니다")
    
    try:
        logger.info(f"검색 요청: {request.user_query}")
        
        # MCP 서버의 search_vector_store 메서드 직접 호출
        results = await mcp_server.search_vector_store(
            user_query=request.user_query,
            database_name=request.database_name,
            vector_store_name=request.vector_store_name,
            k=request.k
        )
        
        logger.info(f"검색 결과: {len(results)}개 항목 반환")
        
        # 자연어 답변 생성
        answer = None
        if request.generate_answer:
            answer = await generate_answer(request.user_query, results)
            logger.info("자연어 답변 생성 완료")
        
        return SearchResponse(
            success=True,
            results=results,
            answer=answer
        )
        
    except Exception as e:
        logger.error(f"검색 중 오류 발생: {e}")
        return SearchResponse(
            success=False,
            results=[],
            answer=None,
            error=str(e)
        )

@app.get("/databases")
async def list_databases():
    """데이터베이스 목록 조회"""
    if not mcp_server:
        raise HTTPException(status_code=500, detail="MCP 서버가 초기화되지 않았습니다")
    
    try:
        databases = await mcp_server.list_databases()
        return {"success": True, "databases": databases}
    except Exception as e:
        logger.error(f"데이터베이스 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/databases/{database_name}/vector-stores")
async def list_vector_stores(database_name: str):
    """벡터 스토어 목록 조회"""
    if not mcp_server:
        raise HTTPException(status_code=500, detail="MCP 서버가 초기화되지 않았습니다")
    
    try:
        vector_stores = await mcp_server.list_vector_stores(database_name)
        return {"success": True, "vector_stores": vector_stores}
    except Exception as e:
        logger.error(f"벡터 스토어 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "rest_api_wrapper:app",
        host="0.0.0.0",
        port=3002,  # 3001과 다른 포트 사용
        reload=False,
        log_level="info"
    )