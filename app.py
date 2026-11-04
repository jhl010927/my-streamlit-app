
import os, base64, json, requests
import streamlit as st
from openai import OpenAI

# OpenAI 클라이언트 생성
client = OpenAI()

# 페이지 설정
st.set_page_config(page_title="음성 AI 서비스", page_icon="🤖", layout="wide")
st.title("음성 인식 AI 서비스")
st.write("음성으로 질문하시면 답변해 드립니다.")

# 역할 지정
sys_role = """
당신은 다양한 업무를 처리하는 AI 서비스입니다.
개인 비서처럼 친절하게 응대합니다.
"""

# 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 지시 영역
with st.form("task_form"):
    uploaded_file = st.file_uploader("텍스트 파일 업로드", type=["txt"])
    audio_value = st.audio_input("무엇을 도와드릴까요?")
    submit = st.form_submit_button("실행")

# 버튼을 눌렀을 때만 실행
if submit:
    # 업로드 된 파일이 있으면 디코딩
    text = ""
    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")

    # 녹음이 없으면 녹음 요청
    if audio_value is None:
        st.warning("음성을 녹음해주세요.")
        st.stop()

    # 시스템 메세지
    system_content = sys_role
    if text:
        system_content += f"\n\n다음 내용을 참고하세요:\n{text}"
    input_messages = [{"role": "system", "content": system_content}]

    # STT: 음성 → 텍스트
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=("audio.wav", audio_value, "audio/wav"))

    # 사용자 메시지
    input_messages.append({"role": "user", "content": transcript.text})

    # LLM 호출 및 응답 듣기
    with st.spinner("답변을 준비하고 있습니다..."):
        # LLM 호출
        response = client.responses.create(
            model="gpt-4o-mini",
            input=input_messages
        )
        answer = response.output_text
  # 회의 내용을 화이트보드 항목으로 받기 위한 출력 형식
board_schema = {
    "type": "json_schema",
    "name": "meeting_whiteboard",
    "schema": {
        "type": "object",
        "properties": {
            "회의요약": {"type": "string"},
            "논의주제": {"type": "array", "items": {"type": "string"}},
            "결정사항": {"type": "array", "items": {"type": "string"}},
            "업무목록": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "업무": {"type": "string"},
                        "담당자": {"type": ["string", "null"]},
                        "기한": {"type": ["string", "null"]}
                    },
                    "required": ["업무", "담당자", "기한"],
                    "additionalProperties": False
                }
            },
            "미결사항": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["회의요약", "논의주제", "결정사항", "업무목록", "미결사항"],
        "additionalProperties": False
    }
}

# 회의 내용을 구조화된 형식으로 AI에 요청
request["text"] = {"format": board_schema}
response = client.responses.create(**request)
board = json.loads(response.output_text)

# 화이트보드 항목 표시
st.subheader("회의 화이트보드")
st.info(board["회의요약"])

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown("### 논의 주제")
        for item in board["논의주제"]:
            st.markdown(f"- {item}")

    with st.container(border=True):
        st.markdown("### 결정사항")
        for item in board["결정사항"]:
            st.markdown(f"- {item}")

with col2:
    with st.container(border=True):
        st.markdown("### 해야 할 일")
        st.table(board["업무목록"])

    with st.container(border=True):
        st.markdown("### 미결사항")
        for item in board["미결사항"]:
            st.markdown(f"- {item}")

# 정리된 회의 내용을 화이트보드 PNG로 생성
board_for_image = json.dumps(board, ensure_ascii=False, indent=2)

image_prompt = f"""
회의 내용을 정리한 화이트보드 사진을 만들어 주세요.
깔끔한 회의실 화이트보드에 제목과 항목을 보기 좋게 배치하세요.
아래 한국어 내용을 바꾸거나 새로 추가하지 마세요.

{board_for_image}
"""

image_response = client.images.generate(
    model="gpt-image-1-mini",
    prompt=image_prompt,
    size="1024x1024"
)

image_bytes = base64.b64decode(image_response.data[0].b64_json)

st.image(image_bytes, caption="AI가 생성한 회의 화이트보드")
st.download_button(
    "화이트보드 PNG 다운로드",
    data=image_bytes,
    file_name="회의_화이트보드.png",
    mime="image/png"
)

        # 응답 출력
        st.subheader("AI 답변")
        st.write(answer)
