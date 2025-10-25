# langchain_mcp_server.py - FIXED + Team Info
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

app = FastAPI(title="LangChain + MCP Vector Chatbot", version="3.1.0")

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
            
            documents = []
            
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    # ========== 1. 아이템 데이터 로드 ==========
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
                    
                    # ========== 2. 팀 정보 데이터 로드 ==========
                    await cursor.execute("""
                        SELECT name, role, emoji
                        FROM team_members
                        ORDER BY id
                    """)
                    team_members = await cursor.fetchall()
            
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
            
            # ========== 아이템 Document 생성 ==========
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
                        "source": "items_table",
                        "type": "item"
                    }
                )
                documents.append(doc)
            
            # ========== 팀 정보 Document 생성 ==========
            if team_members:
                team_content = "🤝 개발팀 정보\n\n"
                for member in team_members:
                    name = member[0]
                    role = member[1]
                    emoji = member[2] or ""
                    team_content += f"{emoji} {name}: {role}\n"
                
                team_doc = Document(
                    page_content=team_content,
                    metadata={
                        "source": "team_members",
                        "type": "team_info"
                    }
                )
                documents.append(team_doc)
                logger.info(f"팀 정보 문서 추가: {len(team_members)}명")
            
            self.documents_cache = documents
            
            # FAISS 벡터 스토어 생성
            if documents:
                self.vector_store = FAISS.from_documents(
                    documents,
                    self.embeddings
                )
                logger.info(f"벡터 스토어 생성 완료: {len(documents)}개 문서 (아이템 + 팀 정보)")
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
            pool = self.mcp_server.pool
            documents = []
            
            # 팀 정보 관련 키워드 체크
            team_keywords = ['개발자', '팀', '멤버', '구성원', '누가', '누구', '만든', '제작']
            is_team_query = any(keyword in query for keyword in team_keywords)
            
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    # 팀 정보 검색
                    if is_team_query:
                        await cursor.execute("""
                            SELECT name, role, emoji
                            FROM team_members
                            ORDER BY id
                        """)
                        team_results = await cursor.fetchall()
                        
                        if team_results:
                            team_content = "🤝 개발팀 정보\n\n"
                            for member in team_results:
                                name = member[0]
                                role = member[1]
                                emoji = member[2] or ""
                                team_content += f"{emoji} {name}: {role}\n"
                            
                            doc = Document(
                                page_content=team_content,
                                metadata={
                                    "source": "team_members",
                                    "type": "team_info"
                                }
                            )
                            documents.append(doc)
                    
                    # 아이템 검색
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
                            metadata={
                                "item_id": row[0],
                                "name": row[1],
                                "rarity": row[2],
                                "source": "direct_sql",
                                "type": "item"
                            }
                        )
                        documents.append(doc)
            
            return documents
        except Exception as e:
            logger.error(f"직접 SQL 검색 오류: {e}")
            return []

# ==================== 챗봇 서비스 ====================
class SmartChatService:
    """통합 챗봇 서비스"""
    
    def __init__(self, mcp_server: MariaDBServer, vector_service: VectorSearchService):
        self.mcp_server = mcp_server
        self.vector_service = vector_service
        self.sql_searcher = DirectSQLSearcher(mcp_server)
        
        self.chat_model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7
        )
        
        self.prompt_template = PromptTemplate.from_template("""
당신은 게임 아이템 정보와 개발팀 정보를 안내하는 친절한 AI 어시스턴트입니다.

[대화 기록]
{history}

[검색된 관련 정보]
{context}

[사용자 질문]
{input}

위 정보를 바탕으로 사용자의 질문에 친절하고 정확하게 답변해주세요.
- 아이템 정보 질문이면 아이템 이름, 등급, 능력치 등을 설명해주세요.
- 개발팀/개발자 정보 질문이면 팀 구성원과 역할을 안내해주세요.
- 검색 결과에 없는 내용은 추측하지 마세요.
- 이모지를 적절히 사용하여 답변을 보기 좋게 작성하세요.

답변:
""")
    
    async def generate_response(
        self,
        user_query: str,
        session_id: str = "default",
        use_vector: bool = True
    ) -> Dict[str, Any]:
        """사용자 질문에 대한 응답 생성"""
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
                    "answer": "죄송합니다. 해당 정보를 찾을 수 없습니다. 다른 키워드로 검색해보세요!",
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
            sources = []
            for doc in relevant_docs[:3]:
                if doc.metadata.get("type") == "team_info":
                    sources.append({
                        "type": "team_info",
                        "source": "team_members"
                    })
                else:
                    sources.append({
                        "item_id": doc.metadata.get("item_id"),
                        "name": doc.metadata.get("name_ko"),
                        "rarity": doc.metadata.get("rarity"),
                        "source": doc.metadata.get("source"),
                        "type": "item"
                    })
            
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