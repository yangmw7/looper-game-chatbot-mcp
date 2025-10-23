# langchain_mcp_server.py - FIXED
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import logging
from typing import List, Dict, Any, Optional
import uvicorn
import os
from datetime import datetime
import sys

# Windows 콘솔 인코딩 설정
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# LangChain imports
from langchain.schema import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

# MCP 서버
from server import MariaDBServer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="LangChain + MCP Vector Chatbot", version="3.0.0")

# 전역 변수
mcp_server: Optional[MariaDBServer] = None
vector_service: Optional['VectorSearchService'] = None

# 세션별 대화 기록
conversation_memories: Dict[str, List[Dict[str, str]]] = {}

class ChatRequest(BaseModel):
    user_query: str
    session_id: str = "default"
    use_vector_search: bool = True

class ChatResponse(BaseModel):
    success: bool
    answer: str
    search_method: str
    sources: List[Dict[str, Any]] = []
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    message: str
    timestamp: str
    services: Dict[str, str]
    active_sessions: int
    vector_store_initialized: bool

# ==================== 벡터 검색 서비스 ====================
class VectorSearchService:
    """MCP + LangChain 벡터 검색 서비스"""
    
    def __init__(self, mcp_server: MariaDBServer):
        self.mcp_server = mcp_server
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.documents_cache = []
        
    async def initialize_vector_store(self):
        """MCP로 DB 데이터 가져와서 벡터 스토어 생성"""
        try:
            logger.info("벡터 스토어 초기화 시작...")
            
            # 🔥 수정: self.mcp_server.pool 직접 사용
            pool = self.mcp_server.pool
            if not pool:
                raise Exception("DB 연결 풀이 초기화되지 않았습니다")
            
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    # items와 item_names, item_descriptions 조인
                    await cursor.execute("""
                        SELECT 
                            i.id,
                            i.rarity,
                            i.`two-hander`,
                            i.stackable,
                            GROUP_CONCAT(DISTINCT CASE WHEN n.lang = 'ko' THEN n.value END) as name_ko,
                            GROUP_CONCAT(DISTINCT CASE WHEN n.lang = 'en' THEN n.value END) as name_en,
                            GROUP_CONCAT(DISTINCT CASE WHEN d.lang = 'ko' THEN d.value END) as desc_ko,
                            GROUP_CONCAT(DISTINCT CASE WHEN d.lang = 'en' THEN d.value END) as desc_en
                        FROM items i
                        LEFT JOIN item_names n ON i.id = n.item_id
                        LEFT JOIN item_descriptions d ON i.id = d.item_id
                        GROUP BY i.id, i.rarity, i.`two-hander`, i.stackable
                    """)
                    items = await cursor.fetchall()
                    
                    # item_attributes 가져오기
                    await cursor.execute("""
                        SELECT item_id, op, stat, value
                        FROM item_attributes
                    """)
                    attributes = await cursor.fetchall()
            
            # 아이템별 속성 매핑
            item_attrs = {}
            for attr in attributes:
                item_id = attr[0]
                if item_id not in item_attrs:
                    item_attrs[item_id] = []
                item_attrs[item_id].append({
                    'op': attr[1],
                    'stat': attr[2],
                    'value': attr[3]
                })
            
            # Document 생성
            documents = []
            for item in items:
                item_id = item[0]
                name_ko = item[4] or "이름 없음"
                name_en = item[5] or "No name"
                desc_ko = item[6] or ""
                desc_en = item[7] or ""
                rarity = item[1]
                two_hander = item[2]
                stackable = item[3]
                
                # 속성 정보
                attrs = item_attrs.get(item_id, [])
                attrs_text = ", ".join([f"{a['stat']}: {a['op']} {a['value']}" for a in attrs])
                
                # 검색 가능한 텍스트 구성
                content = f"""
아이템 ID: {item_id}
한글 이름: {name_ko}
영문 이름: {name_en}
등급: {rarity}
양손 무기: {'예' if two_hander else '아니오'}
중첩 가능: {'예' if stackable else '아니오'}
설명(한글): {desc_ko}
설명(영문): {desc_en}
능력치: {attrs_text}
                """.strip()
                
                doc = Document(
                    page_content=content,
                    metadata={
                        "item_id": item_id,
                        "name_ko": name_ko,
                        "name_en": name_en,
                        "rarity": rarity,
                        "attributes": attrs,
                        "source": "items_table"
                    }
                )
                documents.append(doc)
            
            self.documents_cache = documents
            
            # FAISS 벡터 스토어 생성
            if documents:
                self.vector_store = FAISS.from_documents(
                    documents,
                    self.embeddings
                )
                logger.info(f"벡터 스토어 생성 완료: {len(documents)}개 문서")
            else:
                logger.warning("문서가 없어서 벡터 스토어를 생성하지 못했습니다")
                
        except Exception as e:
            logger.error(f"벡터 스토어 초기화 실패: {e}")
            raise
    
    async def search_similar(self, query: str, k: int = 5) -> List[Document]:
        """의미 기반 유사도 검색"""
        if not self.vector_store:
            logger.warning("벡터 스토어가 초기화되지 않았습니다")
            return []
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"벡터 검색 결과: {len(results)}개")
            return results
        except Exception as e:
            logger.error(f"벡터 검색 오류: {e}")
            return []

# ==================== 직접 SQL 검색 (Fallback) ====================
class DirectSQLSearcher:
    """벡터 검색 실패 시 사용하는 직접 SQL 검색"""
    
    def __init__(self, mcp_server: MariaDBServer):
        self.mcp_server = mcp_server
    
    async def search(self, query: str, k: int = 3) -> List[Document]:
        """간단한 LIKE 검색"""
        try:
            # 🔥 수정: self.mcp_server.pool 직접 사용
            pool = self.mcp_server.pool
            documents = []
            
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute("""
                        SELECT DISTINCT i.id, n.value, i.rarity
                        FROM items i
                        JOIN item_names n ON i.id = n.item_id
                        WHERE n.value LIKE %s AND n.lang = 'ko'
                        LIMIT %s
                    """, (f"%{query}%", k))
                    
                    results = await cursor.fetchall()
                    
                    for row in results:
                        content = f"아이템: {row[1]} (ID: {row[0]}, 등급: {row[2]})"
                        doc = Document(
                            page_content=content,
                            metadata={"item_id": row[0], "name": row[1], "rarity": row[2], "source": "direct_sql"}
                        )
                        documents.append(doc)
            
            logger.info(f"직접 SQL 검색 결과: {len(documents)}개")
            return documents
            
        except Exception as e:
            logger.error(f"직접 SQL 검색 오류: {e}")
            return []

# ==================== 대화형 챗봇 서비스 ====================
class SmartChatService:
    """벡터 검색과 직접 SQL을 모두 활용하는 스마트 챗봇"""
    
    def __init__(self, mcp_server: MariaDBServer, vector_service: VectorSearchService):
        self.mcp_server = mcp_server
        self.vector_service = vector_service
        self.sql_searcher = DirectSQLSearcher(mcp_server)
        self.chat_model = ChatOpenAI(
            temperature=0.3,
            model="gpt-4o",
            max_tokens=600,
            timeout=30
        )
        
        self.prompt_template = PromptTemplate(
            input_variables=["history", "context", "input"],
            template="""당신은 'Looper RPG' 게임의 정확한 게임 가이드 AI입니다.

📋 답변 규칙:
1. 데이터베이스에 있는 정보만 사용 (절대 추측 금지)
2. 아이템명, 등급, 능력치 등 구체적 정보 제공
3. 한글과 영문 이름 모두 언급
4. 능력치는 정확한 수치로 표시

이전 대화:
{history}

📊 검색된 게임 데이터:
{context}

❓ 사용자 질문: {input}

위 데이터를 바탕으로 정확하게 답변해주세요."""
        )
    
    async def generate_response(
        self, 
        user_query: str, 
        session_id: str,
        use_vector: bool = True
    ) -> Dict[str, Any]:
        """응답 생성"""
        try:
            # 1. 검색 방법 선택
            if use_vector and self.vector_service.vector_store:
                search_method = "vector"
                relevant_docs = await self.vector_service.search_similar(user_query, k=5)
            else:
                search_method = "direct_sql"
                relevant_docs = await self.sql_searcher.search(user_query, k=3)
            
            # 2. 검색 결과 없으면 실패
            if not relevant_docs:
                return {
                    "answer": "죄송합니다. 해당 아이템에 대한 정보를 찾을 수 없습니다. 다른 키워드로 검색해보세요!",
                    "search_method": search_method,
                    "sources": []
                }
            
            # 3. 컨텍스트 구성
            context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs[:3]])
            
            # 4. 대화 기록 가져오기
            history = ""
            if session_id in conversation_memories:
                recent = conversation_memories[session_id][-4:]
                history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent])
            
            # 5. 프롬프트 생성
            prompt = self.prompt_template.format(
                history=history,
                context=context,
                input=user_query
            )
            
            # 6. AI 응답 생성
            response = await self.chat_model.ainvoke(prompt)
            answer = response.content
            
            # 7. 대화 기록 저장
            if session_id not in conversation_memories:
                conversation_memories[session_id] = []
            
            conversation_memories[session_id].extend([
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": answer}
            ])
            
            # 최근 10턴만 유지
            if len(conversation_memories[session_id]) > 20:
                conversation_memories[session_id] = conversation_memories[session_id][-20:]
            
            # 8. 소스 정보 구성
            sources = [
                {
                    "item_id": doc.metadata.get("item_id"),
                    "name": doc.metadata.get("name_ko"),
                    "rarity": doc.metadata.get("rarity"),
                    "source": doc.metadata.get("source")
                }
                for doc in relevant_docs[:3]
            ]
            
            return {
                "answer": answer,
                "search_method": search_method,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"응답 생성 오류: {e}")
            return {
                "answer": "죄송합니다. 응답 생성 중 오류가 발생했습니다.",
                "search_method": "error",
                "sources": []
            }

# ==================== FastAPI 이벤트 ====================
chat_service: Optional[SmartChatService] = None

@app.on_event("startup")
async def startup_event():
    """서버 시작"""
    global mcp_server, vector_service, chat_service
    
    try:
        logger.info("=" * 60)
        logger.info("벡터 검색 기반 챗봇 서버 시작")
        logger.info("=" * 60)
        
        # 1. MCP 서버 초기화
        logger.info("MCP 서버 초기화...")
        mcp_server = MariaDBServer()
        await mcp_server.initialize_pool()
        mcp_server.register_tools()
        logger.info("MCP 서버 초기화 완료")
        
        # 2. 벡터 검색 서비스 초기화
        logger.info("벡터 검색 서비스 초기화...")
        vector_service = VectorSearchService(mcp_server)
        await vector_service.initialize_vector_store()
        logger.info("벡터 검색 서비스 초기화 완료")
        
        # 3. 챗봇 서비스 초기화
        logger.info("챗봇 서비스 초기화...")
        chat_service = SmartChatService(mcp_server, vector_service)
        logger.info("챗봇 서비스 초기화 완료")
        
        # 4. OpenAI API 키 확인
        if not os.getenv('OPENAI_API_KEY'):
            logger.warning("OPENAI_API_KEY 환경변수가 설정되지 않았습니다!")
        else:
            logger.info("OpenAI API 키 확인됨")
        
        logger.info("=" * 60)
        logger.info("시스템 준비 완료!")
        logger.info("서버: http://0.0.0.0:3003")
        logger.info("Chat: POST http://localhost:3003/chat")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"초기화 실패: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료"""
    global mcp_server
    if mcp_server:
        await mcp_server.close_pool()
        logger.info("서버 종료 완료")

# ==================== API 엔드포인트 ====================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """헬스 체크"""
    openai_status = "UP" if os.getenv('OPENAI_API_KEY') else "DOWN"
    mcp_status = "UP" if mcp_server else "DOWN"
    vector_status = "UP" if (vector_service and vector_service.vector_store) else "DOWN"
    chat_status = "UP" if chat_service else "DOWN"
    
    overall = "UP" if all([
        openai_status == "UP",
        mcp_status == "UP",
        vector_status == "UP",
        chat_status == "UP"
    ]) else "DEGRADED"
    
    return HealthResponse(
        status=overall,
        message="Vector Search Chatbot is running",
        timestamp=datetime.now().isoformat(),
        services={
            "openai": openai_status,
            "mcp_server": mcp_status,
            "vector_store": vector_status,
            "chat_service": chat_status
        },
        active_sessions=len(conversation_memories),
        vector_store_initialized=(vector_service and vector_service.vector_store is not None)
    )

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """채팅 엔드포인트"""
    if not chat_service:
        raise HTTPException(status_code=500, detail="챗봇 서비스가 초기화되지 않았습니다")
    
    try:
        logger.info(f"질문 (세션: {request.session_id}): {request.user_query}")
        
        result = await chat_service.generate_response(
            user_query=request.user_query,
            session_id=request.session_id,
            use_vector=request.use_vector_search
        )
        
        logger.info(f"응답 완료 (방법: {result['search_method']})")
        
        return ChatResponse(
            success=True,
            answer=result["answer"],
            search_method=result["search_method"],
            sources=result.get("sources", [])
        )
        
    except Exception as e:
        logger.error(f"오류: {e}")
        return ChatResponse(
            success=False,
            answer="죄송합니다. 오류가 발생했습니다.",
            search_method="error",
            error=str(e)
        )

@app.post("/vector/rebuild")
async def rebuild_vector_store():
    """벡터 스토어 재구축"""
    if not vector_service:
        raise HTTPException(status_code=500, detail="벡터 서비스가 없습니다")
    
    try:
        await vector_service.initialize_vector_store()
        return {"message": "벡터 스토어가 재구축되었습니다", "document_count": len(vector_service.documents_cache)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"재구축 실패: {e}")

@app.get("/sessions")
async def list_sessions():
    """세션 목록"""
    return {
        "total_sessions": len(conversation_memories),
        "sessions": list(conversation_memories.keys())
    }

@app.delete("/chat/{session_id}")
async def clear_session(session_id: str):
    """세션 삭제"""
    if session_id in conversation_memories:
        del conversation_memories[session_id]
        return {"message": f"세션 {session_id} 삭제됨"}
    return {"message": "세션이 존재하지 않습니다"}

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=3003,
        reload=False,
        log_level="info"
    )