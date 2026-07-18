import streamlit as st

from src.inference import run_meeting_analysis
from src.parser import parse_meeting_text, is_vtt_transcript
from src.schemas import Participant

st.set_page_config(
    page_title = "Meeting Attribution Notes",
    layout = "wide",
)

st.title("Meeting Attribution Notes")
st.caption("Upload or paste a transcript to have attributed speakers and "
           "notes generated.")

st.sidebar.header("Participants:")
st.sidebar.caption("Add everyone who was in the meeting. "
                   "This will be used to attribute speakers to notes.")

# demo
p1_name = st.sidebar.text_input("Participant 1 Name", value = "Alice McAliceFace")
p1_role = st.sidebar.text_input("Participant 1 Role", value = "design")
p1_dept = st.sidebar.text_input("Participant 1 Department", value = "product")

p2_name = st.sidebar.text_input("Participant 2 Name", value = "Bob McBobFace")
p2_role = st.sidebar.text_input("Participant 2 Role", value = "finance")
p2_dept = st.sidebar.text_input("Participant 2 Department", value = "finance")

st.subheader("Transcript:")

# demo - built in samples 
sample_choice = st.selectbox(
    "Load a sample (optional):",
    ["None", "Simple (Speaker 1, Speaker 2)", "Zoom VTT", "Teams VTT"],
)

uploaded = st.file_uploader(
    "Or upload a .txt/.vtt file",
    type = ["txt", "vtt"],
)

transcript_text = st.text_area(
    "Or paste your transcript here:",
    height = 300,
    placeholder = "Paste your transcript here...",
)

title_override = st.text_input(
    "Meeting title (optional — auto-detected for simple format, defaults for VTT)",
    value="",
)

if sample_choice == "Simple (Speaker 1, Speaker 2)":
    transcript_text = open("data/sample_transcript.txt", encoding = "utf-8").read()
elif sample_choice == "Zoom VTT":
    transcript_text = open("data/sample_transcript_zoom.txt", encoding = "utf-8").read()
elif sample_choice == "Teams VTT":
    transcript_text = open("data/sample_transcript_teams.txt", encoding = "utf-8").read()

if uploaded:
    transcript_text = uploaded.read().decode("utf-8")

analyze_clicked = st.button("Analyze Meeting", type="primary")
if analyze_clicked:
    if not transcript_text.strip():
        st.error("Please paste or upload a transcript to analyze.")
        st.stop()
    
    if not p1_name.strip() or not p2_name.strip():
        st.error("Please add both participants to analyze.")
        st.stop()

    participants = [
        Participant(
            name = p1_name.strip(),
            role = p1_role.strip() or None,
            dept = p1_dept.strip() or None,
        ),
        Participant(
            name = p2_name.strip(),
            role = p2_role.strip() or None,
            dept = p2_dept.strip() or None,
        ),
    ]

    meeting = parse_meeting_text(
        transcript_text,
        participants = participants,
        title = title_override.strip() or None,
    )

    # show status
    format_label = "VTT (Zoom/Teams)" if is_vtt_transcript(transcript_text) else "Plain"
    st.info(f"Parsed as **{ format_label }** - **{ len(meeting.lines) }** lines found.")

    with st.spinner("Calling HuggingFace model (this may take some time)"):
        result = run_meeting_analysis(meeting)

    st.success("Analysis complete")

    # display results
    st.subheader("Attributed Lines")
    for line in result.attributed_lines:
        st.markdown(
            f"**{line.confidence:.2f}%**: **{ line.speaker_name }** { line.text }"
        )
        if line.reasoning:
            st.caption(f"Why: { line.reasoning }")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Summary")
        for bullet in result.summary_bullets:
            st.markdown(f"- { bullet }")

    with col2:
        st.subheader("Action Items")
        for item in result.action_items:
            st.markdown(f"- { item }")