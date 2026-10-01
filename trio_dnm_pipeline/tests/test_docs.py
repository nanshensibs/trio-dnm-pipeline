"""Guards on the shipped protocol document (stdlib only)."""
import os
import re
import zipfile

DOCX = os.path.join(os.path.dirname(__file__), "..", "docs", "Trio_DNM_Pipeline_Protocol_v2.0.docx")


def test_docx_styles_unique_and_default_paragraph_style():
    with zipfile.ZipFile(DOCX) as z:
        styles = z.read("word/styles.xml").decode()
    ids = re.findall(r'w:styleId="([^"]+)"', styles)
    assert len(ids) == len(set(ids)), "duplicate style ids"
    defaults = re.findall(r'<w:style w:type="paragraph"[^>]*w:default="1"', styles)
    assert len(defaults) == 1


def test_docx_states_corrected_thresholds():
    with zipfile.ZipFile(DOCX) as z:
        text = re.sub(r"<[^>]+>", "", z.read("word/document.xml").decode())
    assert "0.016" in text and "SBS40a" in text and "AF_grpmax_joint" in text
    assert "AF_joint_grpmax" not in text
