import html
import streamlit.components.v1 as components

from src.schemas import MeetingInput, MeetingOutput

def render_synced_comparison(
    meeting: MeetingInput,
    result: MeetingOutput,
    height: int = 400,
) -> None:
    """
    Side-by-side comparison of original transcript and attributed lines.
    Has alternating color rows and synced scroll.
    Rows are aligned by index, e.g., Line 1 of the original is aligned with 
    line 1 of the attributed.
    """
    original_lines = meeting.lines
    attributed_lines = result.attributed_lines
    num_rows = min(len(original_lines), len(attributed_lines))

    row_html_parts: list[str] = []

    for i in range(num_rows):
        og = original_lines[i]
        att = attributed_lines[i]

        # left side
        if og.raw_label:
            left_text = f"{ og.raw_label }: { og.text }"
        else:
            left_text = og.text

        # right side — main attribution + reasoning on its own line (same zebra row)
        conf_pct = int(round(att.confidence * 100))
        right_main = (
            f'<div class="attribution-line">'
            f"[{conf_pct}%] <strong>{html.escape(att.speaker_name)}</strong>: "
            f"{html.escape(att.text)}"
            f"</div>"
        )
        reasoning_block = ""
        if att.reasoning:
            reasoning_block = (
                f'<div class="reasoning"><em>{html.escape(att.reasoning)}</em></div>'
            )

        stripe = "even" if i % 2 == 0 else "odd"

        row_html_parts.append(
            f"""
            <div class="row {stripe}">
                <div class="cell left">{html.escape(left_text)}</div>
                <div class="cell right">{right_main}{reasoning_block}</div>
            </div>
            """
        )

    rows_html = "\n".join(row_html_parts)

    html_doc = f"""
    <style>
        .compare-wrap {{
            font-family: sans-serif;
            font-size: 14px;
        }}
        .headers {{
            display: flex;
            font-weight: 700;
            font-size: 15px;
            padding: 10px 8px;
            background: #2b2b2b;
            color: #ffffff;
            border-radius: 6px 6px 0 0;
        }}
        .headers div {{
            flex: 1;
            padding: 0 8px;
        }}
        .scroll-panel {{
            height: { height }px;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 6px;
        }}
        .rows {{
            display: flex;
            flex-direction: column;
        }}
        .row {{
            display: flex;
            min-height: 52px;
            align-items: stretch;
        }}
        .row.even {{
            background: #f5f5f5;
        }}
        .row.odd {{ 
            background: #e9e9e9;
        }}
        .cell {{
            flex: 1;
            padding: 10px 8px;
            border-bottom: 1px solid #ddd;
            line-height: 1.4;
        }}
        .attribution-line {{
            line-height: 1.4;
        }}
        .reasoning {{
            margin-top: 8px;
            font-size: 12px;
            color: #555;
            line-height: 1.35;
        }}
        .sync-scroll {{
            height: {height}px;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 0 0 6px 6px;
        }}
    </style>

    <div class="compare-wrap">
        <div class="headers">
            <div>Original transcript (parsed)</div>
            <div>Attributed lines (model)</div>
        </div>
        <div id="sync-scroll" class="sync-scroll">
            <div class="rows">
                { rows_html }
            </div>
        </div>
    </div>
    """

    components.html(html_doc, height=height + 60, scrolling=False)