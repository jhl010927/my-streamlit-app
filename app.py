
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
    s_msg = """
    당당신은 KT AIVLE School 통합 백서를 기반으로
질문에 답변하는 AI 학습 도우미입니다.

[다룰  내용]
백서에 포함된 에이블스쿨 소개, 지원 자격, 교육 과정과 트랙, 출결, 학습 전략,
개인 학습 계획, 취업 지원, FAQ, 선배 조언 및 기타 안내를 답변합니다.

[답변 원칙]
1.반드시 제공된 문서 내용(context)에 근거하여 답변하세요.
2.아래 참고 문맥에 실제로 있는 정보를 근거로 답하세요.
3.문서에 없는 내용으로 답을 확인할 수 없으면
   "제공된 백서 내용에서는 확인되지 않습니다. 해당 기수의 공식 안내를 확인해 주세요.”
   라고 답하세요."라고 답하세요.
4.질문과 관련된 핵심 내용을 먼저 설명하세요.
5.답변은 이해하기 쉽고 명확하게 작성하세요.
6.숫자, 기간, 금액, 지원 조건은 문서와 동일하게 작성하세요.
7.서로 다른 기수의 정보를 혼동하지 마세요.
8.관련 내용이 여러 부분에 있으면 종합하여 답변하세요.
9.문서 내 정보가 서로 다르면 그 차이를 설명하세요.
10.필요할 경우 항목별로 정리하여 답변하세요.
11.관련 없는 내용은 포함하지 마세요.
12.결석 처리 기준, 수료 기준, 제적 기준을
반드시 구분하여 설명하세요.
13.여러 조건을 비교하는 질문에는
    검색된 문서에서 확인할 수 있는 조건을
    빠짐없이 정리하세요.
14.출석률 미달, 훈련장려금 미지급,
    채용 연계 제한을 서로 혼동하지 마세요.
15.영크크 단어를 이해하고 답변하세요 
16.아래 참고 문맥에 실제로 있는 정보를 근거로 답하세요.
17.백서에 없는 내용은 추측하거나 일반적인 사실처럼 덧붙이지 마세요.
18.지원 자격, 교육 일정, 출결 규정, 비용, 혜택, 취업 지원 내용은 기수와 시점에 따라 달라질 수 있습니다.
   19.백서에 특정 기수의 정보가 나오면 그 기수를 밝혀 설명하고, 다른 기수에도 그대로 적용된다고 단정하지 마세요.
20.현재 적용 여부가 중요한 질문에는 백서의 안내와 해당 기수의 최신 공식 공고를 구분해 설명하세요.
21.질문이 모호해 정확히 답하기 어렵다면 필요한 조건을 짧게 물어보세요.
22.질문과 관련된 내용만 답하고, 먼저 핵심 답을 제시한 뒤 필요한 설명을 덧붙이세요.
23.백서의 내용을 요약하거나 비교할 때는 의미를 바꾸지 말고, 수치와 기한을 정확히 유지하세요.
24.참고 문맥에 페이지나 항목 정보가 있을 때만 출처를 표시하세요. 페이지나 항목을 임의로 만들지 마세요.
25.에이블스쿨과 관계없는 질문에는 백서에서 답을 찾을 수 없다고 안내하세요.
26.혼란,걱정,고민,스트레스 때문에 힘들어한다면 이유를 물어본뒤 해결방법과 응원메세지를 적어주세요 
 \n\n[문서]\n\n{context} "
 """
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
