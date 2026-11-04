
import base64
import json

import streamlit as st
from openai import OpenAI

client = OpenAI()

st.set_page_config(page_title="음성 AI 서비스", page_icon="🤖", layout="wide")
st.title("음성 인식 AI 서비스")
st.write("회의 녹음을 회의록과 화이트보드 이미지로 정리합니다.")

sys_role = """
당신은 회의 내용을 정리하는 AI 비서입니다.
녹음과 입력된 자료에 있는 내용만 사용하고, 확인되지 않은 내용은 추측하지 않습니다.
"""

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
        "required": [
            "회의요약", "논의주제", "결정사항", "업무목록", "미결사항"
        ],
        "additionalProperties": False
    }
}

with st.form("meeting_form"):
    uploaded_file = st.file_uploader("회의 관련 텍스트 파일", type=["txt"])
    audio_value = st.audio_input("회의 내용을 녹음하세요.")
    submit = st.form_submit_button("회의 정리 및 화이트보드 만들기")

if submit:
    if audio_value is None:
        st.warning("회의 내용을 녹음해 주세요.")
        st.stop()

    # 녹음 내용을 글자로 변환
    with st.spinner("녹음 내용을 글자로 바꾸고 있습니다..."):
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=("audio.wav", audio_value.getvalue(), "audio/wav")
        )

    meeting_text = transcript.text

    # 업로드한 텍스트 파일이 있으면 회의 내용에 추가
    if uploaded_file is not None:
        file_text = uploaded_file.getvalue().decode("utf-8", errors="replace")
        meeting_text += "\n\n추가 자료:\n" + file_text

    prompt = f"""
다음 회의 내용을 정리하세요.
회의 요약, 논의 주제, 결정사항, 해야 할 업무, 미결사항을 구분하세요.
업무 담당자나 기한이 나오지 않았다면 null로 작성하세요.

회의 내용:
{meeting_text}
"""

    # 회의 내용을 항목별로 정리
    with st.spinner("회의 내용을 정리하고 있습니다..."):
        response = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": sys_role},
                {"role": "user", "content": prompt}
            ],
            text={"format": board_schema}
        )

    board = json.loads(response.output_text)

    # 회의록 표시
    st.subheader("회의 결과")
    st.write("**회의 요약:**", board["회의요약"])

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
            if board["업무목록"]:
                st.table(board["업무목록"])
            else:
                st.write("추출된 업무가 없습니다.")

        with st.container(border=True):
            st.markdown("### 미결사항")
            for item in board["미결사항"]:
                st.markdown(f"- {item}")

    # 정리 결과를 TXT로 다운로드
    minutes_text = json.dumps(board, ensure_ascii=False, indent=2)
    st.download_button(
        "회의록 TXT 다운로드",
        data=minutes_text,
        file_name="회의록.txt",
        mime="text/plain"
    )

    # 정리된 회의 내용을 화이트보드 이미지로 생성
    image_prompt = f"""
회의 내용을 정리한 화이트보드 사진을 만들어 주세요.
깔끔한 회의실 화이트보드에 제목과 항목을 보기 좋게 배치하세요.
아래 한국어 내용을 바탕으로 만들고, 새로운 내용을 추가하지 마세요.

{minutes_text}
"""

    try:
        with st.spinner("화이트보드 이미지를 만들고 있습니다..."):
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
    except Exception as e:
        st.error(f"화이트보드 이미지 생성에 실패했습니다: {e}")
