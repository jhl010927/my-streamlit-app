
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from operator import itemgetter

# 벡터 저장소 경로, 컬렉션 이름 (Colab에서 만들 때와 같아야 함)
CHROMA_DIR = "chroma_db"
COLLECTION = "aivle_docs"

# 문서 포맷팅 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Embedding 모델, 검색기
@st.cache_resource
def get_retriever():
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=embedding,
        persist_directory=CHROMA_DIR
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# 체인 캐싱
@st.cache_resource
def get_chain():
    retriever = get_retriever()
    s_msg = "당신은 주어진 문서를 기반으로 답변하는 AIVLE School 학습 도우미입니다.\n\n[문서]\n\n{context}
지원자와 교육생이 백서 내용을 이해하고 활용할 수 있도록 정확하고 쉬운 한국어로 설명하세요.
 
[다룰 수 있는 내용]
백서에 포함된 에이블스쿨 소개, 지원 자격, 교육 과정과 트랙, 출결, 학습 전략,
개인 학습 계획, 취업 지원, FAQ, 선배 조언 및 기타 안내를 답변합니다.
 
[답변 기준]
1. 아래 참고 문맥에 실제로 있는 정보를 근거로 답하세요.
2. 백서에 없는 내용은 추측하거나 일반적인 사실처럼 덧붙이지 마세요.
3. 답을 확인할 수 없으면
   “제공된 백서 내용에서는 확인되지 않습니다. 해당 기수의 공식 안내를 확인해 주세요.”
   라고 답하세요.
4. 지원 자격, 교육 일정, 출결 규정, 비용, 혜택, 취업 지원 내용은 기수와 시점에 따라 달라질 수 있습니다.
   백서에 특정 기수의 정보가 나오면 그 기수를 밝혀 설명하고, 다른 기수에도 그대로 적용된다고 단정하지 마세요.
5. 현재 적용 여부가 중요한 질문에는 백서의 안내와 해당 기수의 최신 공식 공고를 구분해 설명하세요.
6. 질문이 모호해 정확히 답하기 어렵다면 필요한 조건을 짧게 물어보세요.
7. 질문과 관련된 내용만 답하고, 먼저 핵심 답을 제시한 뒤 필요한 설명을 덧붙이세요.
8. 백서의 내용을 요약하거나 비교할 때는 의미를 바꾸지 말고, 수치와 기한을 정확히 유지하세요.
9. 참고 문맥에 페이지나 항목 정보가 있을 때만 출처를 표시하세요. 페이지나 항목을 임의로 만들지 마세요.
10. 에이블스쿨과 관계없는 질문에는 백서에서 답을 찾을 수 없다고 안내하세요.
11. 지치고 힘들다고하면 위로해주세요 그리고 희망을 주세요."
    h_msg = "{input}"
    messages = [("system", s_msg), ("human", h_msg)]
    prompt = ChatPromptTemplate.from_messages(messages)
    llm = ChatOpenAI(model="gpt-4o-mini", streaming=True, temperature=0)
    rag_chain = (
        {
            "context": itemgetter("input") | retriever | format_docs,
            "input": itemgetter("input")
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# RAG 체인 준비
chain = get_chain()

# 페이지 설정
st.set_page_config(page_title="AIVLE School 학습 도우미", page_icon="📘")
st.title("📘 AIVLE School 학습 도우미")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 대화 내용 출력
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# 사용자 입력 처리
user_input = st.chat_input("메시지를 입력하세요.")
if user_input:
    # 사용자 메시지 표시
    st.chat_message("user").markdown(user_input)

    # 응답 출력
    with st.chat_message("assistant"):
        response_stream = chain.stream({"input": user_input})
        full_response = st.write_stream(response_stream)

    # 대화 기록 저장
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.session_state.messages.append(AIMessage(content=full_response))
