import base64

import json
import mimetypes
import os
import smtplib
from datetime import date
from email.message import EmailMessage

import streamlit as st
from openai import OpenAI


st.set_page_config(page_title="에이블 AI 개인비서", page_icon="🤖", layout="wide")
st.title("🤖 에이블 AI 개인비서")
st.caption("회의 자료를 정리하고, 결정사항·업무·오늘의 우선순위를 한곳에서 확인합니다.")

SYSTEM_ROLE = """
당신은 직장인의 친절하고 간결한 한국어 AI 개인비서 '에이블'입니다.
사용자가 제공한 회의록, 음성 전사, 이미지, 기존 업무 목록만 근거로 답합니다.
자료에 없는 담당자·날짜·결정·중요도는 만들어내지 않습니다. 알 수 없으면 null 또는 '미정'으로 표시합니다.
제안과 확정된 결정을 구분하고, 서로 다른 자료가 충돌하면 어느 자료의 어떤 내용이 다른지 기록합니다.
이미지 글씨가 불명확하면 추측하지 말고 불확실하다고 표시합니다.
""".strip()

MEETING_SCHEMA = {
    "type": "json_schema",
    "name": "able_meeting_analysis",
    "schema": {
        "type": "object",
        "properties": {
            "회의요약": {"type": "string"},
            "회의주제": {"type": "array", "items": {"type": "string"}},
            "주요논의": {"type": "array", "items": {"type": "string"}},
            "결정사항": {"type": "array", "items": {"type": "string"}},
            "제안사항": {"type": "array", "items": {"type": "string"}},
            "미확정사항": {"type": "array", "items": {"type": "string"}},
            "자료비교": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "구분": {"type": "string", "enum": ["중복", "추가정보", "상충", "판독불확실"]},
                        "내용": {"type": "string"},
                        "텍스트자료": {"type": ["string", "null"]},
                        "이미지자료": {"type": ["string", "null"]}
                    },
                    "required": ["구분", "내용", "텍스트자료", "이미지자료"],
                    "additionalProperties": False
                }
            },
            "업무목록": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "업무": {"type": "string"},
                        "세부내용": {"type": "string"},
                        "담당자": {"type": ["string", "null"]},
                        "기한": {"type": ["string", "null"]},
                        "중요도": {"type": "string", "enum": ["높음", "보통", "낮음", "미정"]},
                        "긴급도": {"type": "string", "enum": ["높음", "보통", "낮음", "미정"]},
                        "우선순위": {"type": "string", "enum": ["높음", "보통", "낮음"]},
                        "판단근거": {"type": "string"},
                        "오늘처리": {"type": "boolean"}
                    },
                    "required": ["업무", "세부내용", "담당자", "기한", "중요도", "긴급도", "우선순위", "판단근거", "오늘처리"],
                    "additionalProperties": False
                }
            },
            "오늘브리핑": {"type": "string"},
            "우선처리추천": {"type": "string"}
        },
        "required": ["회의요약", "회의주제", "주요논의", "결정사항", "제안사항", "미확정사항", "자료비교", "업무목록", "오늘브리핑", "우선처리추천"],
        "additionalProperties": False
    }
}


def get_secret(name):
    """Streamlit Secrets를 먼저 확인하고, 없으면 환경변수를 사용합니다."""
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return value or os.getenv(name)


def get_client():
    """Streamlit Secrets 또는 환경변수의 OPENAI_API_KEY로 클라이언트를 만듭니다."""
    api_key = get_secret("OPENAI_API_KEY")
    return OpenAI(api_key=api_key) if api_key else OpenAI()


def deduplicate_tasks(tasks):
    """업무명과 담당자가 같은 항목은 한 건으로 정리합니다."""
    unique = {}
    for task in tasks:
        key = (" ".join(task.get("업무", "").casefold().split()),
               " ".join((task.get("담당자") or "").casefold().split()))
        if key[0] and key not in unique:
            unique[key] = task
    order = {"높음": 0, "보통": 1, "낮음": 2}
    return sorted(unique.values(), key=lambda item: (not item.get("오늘처리", False), order.get(item.get("우선순위"), 3)))


def format_minutes(result):
    lines = ["[회의 요약]", result["회의요약"], "", "[회의 주제]", *[f"- {x}" for x in result["회의주제"]],
             "", "[주요 논의]", *[f"- {x}" for x in result["주요논의"]], "", "[결정사항]",
             *[f"- {x}" for x in result["결정사항"]], "", "[제안사항]",
             *[f"- {x}" for x in result["제안사항"]], "", "[미확정 사항]",
             *[f"- {x}" for x in result["미확정사항"]], "", "[자료 간 비교]"]
    if result["자료비교"]:
        lines.extend(f"- {x['구분']}: {x['내용']} (텍스트: {x['텍스트자료'] or '없음'} / 이미지: {x['이미지자료'] or '없음'})" for x in result["자료비교"])
    else:
        lines.append("- 확인된 중복·추가·상충 내용이 없습니다.")
    lines.extend(["", "[업무 및 우선순위]"])
    for task in result["업무목록"]:
        lines.append(
            f"- [{task['우선순위']}] {task['업무']} / 담당자: {task['담당자'] or '미정'} / "
            f"기한: {task['기한'] or '미정'} / 중요도: {task['중요도']} / 긴급도: {task['긴급도']}"
        )
        if task["세부내용"]:
            lines.append(f"  세부 내용: {task['세부내용']}")
        if task["판단근거"]:
            lines.append(f"  판단 근거: {task['판단근거']}")
    lines.extend(["", "[오늘의 업무 브리핑]", result["오늘브리핑"], "", "[우선 처리 추천]", result["우선처리추천"]])
    return "\n".join(lines)


def show_api_error(error):
    message = str(error).lower()
    if "401" in message or "authentication" in message or "api key" in message:
        st.error("OpenAI 인증에 실패했습니다. Streamlit Secrets에 OPENAI_API_KEY가 등록되어 있는지 확인해 주세요.")
    elif "429" in message or "rate limit" in message:
        st.error("요청 한도 또는 사용량 제한에 도달했습니다. 잠시 후 다시 시도해 주세요.")
    else:
        st.error(f"요청을 처리하지 못했습니다. 네트워크와 API 설정을 확인해 주세요. ({type(error).__name__})")


def speak_text(client, text, key):
    if not text.strip():
        st.info("읽어드릴 내용이 없습니다.")
        return
    try:
        with st.spinner("음성을 만들고 있습니다..."):
            audio = client.audio.speech.create(
                model="gpt-4o-mini-tts", voice="nova", input=text[:4000], response_format="mp3"
            )
        st.audio(audio.content, format="audio/mp3")
    except Exception as exc:
        show_api_error(exc)


def send_email(recipient, subject, body):
    address = get_secret("GMAIL_ADDRESS") or ""
    app_password = get_secret("GMAIL_APP_PASSWORD") or ""
    if not address or not app_password:
        raise RuntimeError("Streamlit Secrets에 GMAIL_ADDRESS와 GMAIL_APP_PASSWORD를 설정해 주세요.")
    message = EmailMessage()
    message["From"] = address
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as smtp:
        smtp.login(address, app_password)
        smtp.send_message(message)


# 사용자별 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "tasks" not in st.session_state:
    st.session_state.tasks = []
if "latest_minutes" not in st.session_state:
    st.session_state.latest_minutes = ""
if "latest_briefing" not in st.session_state:
    st.session_state.latest_briefing = ""
if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

with st.sidebar:
    st.subheader("에이블 설정")
    st.caption(f"오늘 날짜: {date.today().isoformat()}")
    if st.button("대화 기록 초기화", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    if st.button("업무 목록 초기화", use_container_width=True):
        st.session_state.tasks = []
        st.session_state.latest_result = None
        st.rerun()
    st.caption(f"저장된 업무: {len(st.session_state.tasks)}개 · 최근 대화: 최대 10회")

meeting_tab, briefing_tab, chat_tab, email_tab = st.tabs(["회의 자료 분석", "업무 브리핑", "에이블에게 질문", "이메일 보내기"])

with meeting_tab:
    st.subheader("회의록과 화이트보드·슬라이드를 함께 분석합니다")
    st.write("TXT, 사진, 직접 입력, 녹음 중 필요한 자료만 제공하셔도 됩니다.")
    with st.form("meeting_form"):
        txt_file = st.file_uploader("회의록 TXT 파일", type=["txt"], key="meeting_txt")
        image_files = st.file_uploader("화이트보드·슬라이드 이미지", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="meeting_images")
        typed_notes = st.text_area("회의 내용을 직접 입력할 수 있습니다", height=130)
        audio_input = st.audio_input("회의 내용을 녹음해 추가할 수 있습니다", key="meeting_audio")
        run_analysis = st.form_submit_button("통합 분석 및 업무 우선순위 정리", type="primary")

    if run_analysis:
        text_sources = []
        if txt_file is not None:
            try:
                text_sources.append(("TXT 회의록", txt_file.getvalue().decode("utf-8-sig")))
            except UnicodeDecodeError:
                text_sources.append(("TXT 회의록", txt_file.getvalue().decode("cp949", errors="replace")))
        if typed_notes.strip():
            text_sources.append(("직접 입력", typed_notes.strip()))
        client = None
        if audio_input is not None:
            try:
                client = get_client()
                with st.spinner("음성을 텍스트로 바꾸고 있습니다..."):
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1", file=("meeting.wav", audio_input.getvalue(), "audio/wav")
                    )
                text_sources.append(("음성 전사", transcript.text))
            except Exception as exc:
                show_api_error(exc)
                st.stop()
        if not text_sources and not image_files:
            st.warning("TXT, 이미지, 직접 입력, 녹음 중 하나 이상 추가해 주세요.")
            st.stop()

        today = date.today().isoformat()
        current_tasks = st.session_state.tasks
        user_content = [{
            "type": "input_text",
            "text": (
                f"오늘은 {today}입니다. 아래 회의 자료를 통합 분석하고 JSON 형식으로 답하세요.\n"
                "텍스트와 이미지 사이의 중복, 이미지에만 있는 추가 정보, 상충, 판독 불확실을 구분하세요. "
                "기존 업무를 포함해 업무 우선순위를 다시 계산하되, 기존 정보와 새 자료를 임의로 바꾸지 마세요. "
                "기한이 오늘이거나 오늘 처리해야 한다고 자료에 명시된 업무만 오늘처리=true로 표시하세요. "
                "날짜가 모호하면 기한을 null로 두고 오늘처리는 false로 하세요. 중요도·긴급도 근거가 없으면 미정으로 두세요. "
                "우선순위는 자료에 명시된 기한, 중요도, 긴급도만 이용해 정하고 판단근거를 짧게 쓰세요. "
                "기존 업무 목록은 다음과 같습니다:\n" + json.dumps(current_tasks, ensure_ascii=False) +
                "\n\n자료별 텍스트:\n" + "\n\n".join(f"[{name}]\n{body}" for name, body in text_sources)
            )
        }]
        for image in image_files or []:
            mime = image.type or mimetypes.guess_type(image.name)[0] or "image/jpeg"
            data = base64.b64encode(image.getvalue()).decode("ascii")
            user_content.append({"type": "input_image", "image_url": f"data:{mime};base64,{data}", "detail": "high"})
        try:
            client = client or get_client()
            with st.spinner("회의 자료를 비교하고 업무와 우선순위를 정리하고 있습니다..."):
                response = client.responses.create(
                    model="gpt-4o-mini",
                    input=[{"role": "system", "content": SYSTEM_ROLE}, {"role": "user", "content": user_content}],
                    text={"format": MEETING_SCHEMA}
                )
            result = json.loads(response.output_text)
            result["업무목록"] = deduplicate_tasks(result["업무목록"])
            st.session_state.latest_result = result
            st.session_state.latest_minutes = format_minutes(result)
            st.session_state.latest_briefing = result["오늘브리핑"] + "\n\n우선 처리 추천: " + result["우선처리추천"]
            st.session_state.tasks = result["업무목록"]
        except Exception as exc:
            show_api_error(exc)

    result = st.session_state.latest_result
    if result:
        st.divider()
        st.subheader("통합 회의록")
        st.info(result["회의요약"])
        left, right = st.columns(2)
        with left:
            st.markdown("**회의 주제**")
            st.write("\n".join(f"• {x}" for x in result["회의주제"]) or "없음")
            st.markdown("**주요 논의**")
            st.write("\n".join(f"• {x}" for x in result["주요논의"]) or "없음")
            st.markdown("**결정사항**")
            st.write("\n".join(f"• {x}" for x in result["결정사항"]) or "없음")
            st.markdown("**제안사항**")
            st.write("\n".join(f"• {x}" for x in result["제안사항"]) or "없음")
        with right:
            st.markdown("**미확정 사항**")
            st.write("\n".join(f"• {x}" for x in result["미확정사항"]) or "없음")
            st.markdown("**자료 간 비교**")
            if result["자료비교"]:
                st.dataframe(result["자료비교"], use_container_width=True, hide_index=True)
            else:
                st.write("확인된 중복·추가·상충 내용이 없습니다.")
        st.markdown("**업무와 우선순위**")
        if result["업무목록"]:
            st.dataframe(result["업무목록"], use_container_width=True, hide_index=True)
        else:
            st.write("추출된 업무가 없습니다.")
        st.download_button("회의록·업무 TXT 다운로드", st.session_state.latest_minutes, "able_meeting_minutes.txt", "text/plain; charset=utf-8")
        st.download_button("분석 결과 JSON 다운로드", json.dumps(result, ensure_ascii=False, indent=2), "able_meeting_analysis.json", "application/json")
        if st.button("회의록 음성으로 듣기", key="speak_minutes"):
            speak_text(get_client(), st.session_state.latest_minutes, "minutes")

with briefing_tab:
    st.subheader("오늘 처리할 업무와 우선 업무")
    if st.session_state.tasks:
        tasks = st.session_state.tasks
        today_tasks = [x for x in tasks if x.get("오늘처리")]
        recommended = [x for x in tasks if x.get("우선순위") == "높음"]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 오늘 처리할 업무")
            st.dataframe(today_tasks, use_container_width=True, hide_index=True) if today_tasks else st.write("오늘 처리할 것으로 표시된 업무가 없습니다.")
        with c2:
            st.markdown("#### 우선 처리 추천")
            st.dataframe(recommended, use_container_width=True, hide_index=True) if recommended else st.write("높은 우선순위로 분류된 업무가 없습니다.")
        if st.session_state.latest_briefing:
            st.markdown("#### 브리핑")
            st.write(st.session_state.latest_briefing)
            if st.button("오늘 브리핑 음성으로 듣기"):
                speak_text(get_client(), st.session_state.latest_briefing, "briefing")
    else:
        st.info("회의 자료를 먼저 분석하면 업무 목록과 오늘의 브리핑이 여기에 표시됩니다.")

with chat_tab:
    st.subheader("텍스트 질문과 대화 기록")
    for item in st.session_state.chat_history:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
    question = st.chat_input("에이블에게 질문하세요")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.session_state.chat_history = st.session_state.chat_history[-20:]
        with st.chat_message("user"):
            st.markdown(question)
        try:
            client = get_client()
            messages = [{"role": "system", "content": SYSTEM_ROLE}]
            messages.extend(st.session_state.chat_history[-20:])
            with st.chat_message("assistant"):
                with st.spinner("답변을 준비하고 있습니다..."):
                    response = client.responses.create(model="gpt-4o-mini", input=messages)
                answer = response.output_text
                st.markdown(answer)
                if st.button("답변 음성으로 듣기", key=f"speak_chat_{len(st.session_state.chat_history)}"):
                    speak_text(client, answer, "chat")
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.session_state.chat_history = st.session_state.chat_history[-20:]
        except Exception as exc:
            show_api_error(exc)

with email_tab:
    st.subheader("이메일 발송")
    st.caption("발송 전에 받는 사람과 내용을 확인하고, 승인 체크 후 직접 발송 버튼을 눌러야 전송됩니다.")
    default_email_body = st.session_state.latest_minutes or st.session_state.latest_briefing
    with st.form("email_form"):
        recipient = st.text_input("받는 사람 이메일")
        subject = st.text_input("제목", value="에이블 회의록 및 업무 브리핑")
        email_body = st.text_area("발송 내용", value=default_email_body, height=220)
        approved = st.checkbox("내용과 받는 사람을 확인했으며, 지금 발송하는 데 동의합니다.")
        send_clicked = st.form_submit_button("확인 후 이메일 발송")
    if send_clicked:
        if not approved:
            st.warning("발송하려면 승인 체크를 먼저 해 주세요.")
        elif not recipient.strip() or "@" not in recipient:
            st.warning("받는 사람 이메일 주소를 확인해 주세요.")
        elif not email_body.strip():
            st.warning("발송할 내용이 없습니다.")
        else:
            try:
                send_email(recipient.strip(), subject.strip(), email_body)
                st.success("이메일을 발송했습니다.")
            except Exception as exc:
                st.error(f"이메일 발송에 실패했습니다: {exc}")

st.caption("회의 자료와 업무 목록은 현재 브라우저 세션 안에만 보관됩니다. 세션이 종료되면 저장 내용이 초기화될 수 있습니다.")
