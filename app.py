import streamlit as st

from src.inference import run_meeting_analysis
from src.parser import parse_meeting_text, is_vtt_transcript
from src.schemas import Participant
from src.compare_view import render_synced_comparison

st.set_page_config(
    page_title = "Meeting Attribution Notes",
    layout = "wide",
)

st.title("Meeting Attribution Notes")
st.caption("Upload or paste a transcript to have attributed speakers and "
           "notes generated.")

st.sidebar.header("Participants (Optional):")
st.sidebar.caption("Leave at 0 to infer names only from the transcript. "
                   "Add hints when you know roles or names the model might miss. ")

hint_count = st.sidebar.number_input(
    "How many hints?",
    min_value=0,
    max_value=10,
    value=0,
    step=1,
)

participants = []
for i in range(hint_count):
    with st.sidebar.expander(f"Hint {i + 1}", expanded=(i < 2)):
        name = st.text_input("Name", key=f"hint_name_{i}")
        role = st.text_input("Role (optional)", key=f"hint_role_{i}")
        dept = st.text_input("Department (optional)", key=f"hint_dept_{i}")
        if name.strip():
            participants.append(
                Participant(
                    name=name.strip(),
                    role=role.strip() or None,
                    department=dept.strip() or None,
                )
            )

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
    st.subheader("Compare Transcripts")
    render_synced_comparison(meeting, result, height=420)

    if len(meeting.lines) != len(result.attributed_lines):
        st.warning(
            f"Line count mismatch: found {len(meeting.lines)} in transcript, "
            f"but model returned {len(result.attributed_lines)} lines. "
            "Comparison shows the overlapping rows only."
        )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Summary")
        for bullet in result.summary_bullets:
            st.markdown(f"- { bullet }")

    with col2:
        st.subheader("Action Items")
        for item in result.action_items:
            st.markdown(f"- { item }")