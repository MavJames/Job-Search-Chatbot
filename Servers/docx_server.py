"""
MCP Server for creating properly formatted Word documents (.docx)
"""

import sys
import traceback

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor
from fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("docx-creator")


def add_section_heading(doc: Document, text: str):
    """Add a formatted section heading (compact for one-page layout)"""
    heading = doc.add_paragraph(text)
    heading_run = heading.runs[0]
    heading_run.font.size = Pt(11)
    heading_run.font.bold = True
    heading_run.font.color.rgb = RGBColor(0, 0, 0)
    heading.paragraph_format.space_before = Pt(6)
    heading.paragraph_format.space_after = Pt(4)


def _setup_page_margins(doc: Document, margin_inches: float = 0.5):
    """Set page margins"""
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(margin_inches)
        section.bottom_margin = Inches(margin_inches)
        section.left_margin = Inches(margin_inches)
        section.right_margin = Inches(margin_inches)


def _add_contact_info(doc: Document, contact: dict):
    """Add contact info section"""
    if not contact:
        return

    contact_parts = []
    if "email" in contact:
        contact_parts.append(contact["email"])
    if "phone" in contact:
        contact_parts.append(contact["phone"])
    if "location" in contact:
        contact_parts.append(contact["location"])
    if "linkedin" in contact:
        contact_parts.append(contact["linkedin"])

    if contact_parts:
        contact_para = doc.add_paragraph(" | ".join(contact_parts))
        contact_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        contact_para.runs[0].font.size = Pt(9)
        contact_para.paragraph_format.space_after = Pt(6)


def _add_experience_section(doc: Document, experience: list):
    """Add experience section"""
    if not experience:
        return

    add_section_heading(doc, "EXPERIENCE")
    for i, exp in enumerate(experience):
        # Job title and company (bold)
        title_para = doc.add_paragraph()
        title_para.paragraph_format.space_after = Pt(1)
        title_run = title_para.add_run(
            f"{exp.get('title', '')} - {exp.get('company', '')}"
        )
        title_run.bold = True
        title_run.font.size = Pt(10)

        # Location and dates (italic)
        details_para = doc.add_paragraph()
        details_para.paragraph_format.space_after = Pt(2)
        details_run = details_para.add_run(
            f"{exp.get('location', '')} | {exp.get('dates', '')}"
        )
        details_run.italic = True
        details_run.font.size = Pt(9)

        # Responsibilities (bullets, compact)
        if "responsibilities" in exp:
            for resp in exp["responsibilities"]:
                bullet_para = doc.add_paragraph(resp, style="List Bullet")
                bullet_para.paragraph_format.space_after = Pt(1)
                bullet_para.paragraph_format.line_spacing = 1.0
                for run in bullet_para.runs:
                    run.font.size = Pt(10)

        # Space between jobs (smaller)
        if i < len(experience) - 1:
            space_para = doc.add_paragraph()
            space_para.paragraph_format.space_after = Pt(4)


def _add_education_section(doc: Document, education: list):
    """Add education section"""
    if not education:
        return

    add_section_heading(doc, "EDUCATION")
    for edu in education:
        edu_para = doc.add_paragraph()
        edu_para.paragraph_format.space_after = Pt(1)
        degree_run = edu_para.add_run(edu.get("degree", ""))
        degree_run.bold = True
        degree_run.font.size = Pt(10)

        school_para = doc.add_paragraph(
            f"{edu.get('school', '')} - {edu.get('location', '')} | {edu.get('graduation', '')}"
        )
        school_para.paragraph_format.space_after = Pt(4)
        school_para.runs[0].font.size = Pt(9)


@mcp.tool()
def create_resume(
    output_path: str,
    name: str,
    contact: dict = None,
    summary: str = None,
    experience: list = None,
    education: list = None,
    skills: list = None,
) -> str:
    """
    Create a professionally formatted resume in .docx format (optimized for one page)
    """
    try:
        doc = Document()
        _setup_page_margins(doc, 0.5)  # Tight margins

        # Name
        name_para = doc.add_paragraph(name)
        name_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        name_run = name_para.runs[0]
        name_run.font.size = Pt(16)
        name_run.font.bold = True
        name_para.paragraph_format.space_after = Pt(2)

        _add_contact_info(doc, contact)

        # Summary
        if summary:
            add_section_heading(doc, "PROFESSIONAL SUMMARY")
            summary_para = doc.add_paragraph(summary)
            summary_para.paragraph_format.space_after = Pt(6)
            for run in summary_para.runs:
                run.font.size = Pt(10)

        _add_education_section(doc, education)
        _add_experience_section(doc, experience)

        # Skills
        if skills:
            add_section_heading(doc, "SKILLS")
            skills_para = doc.add_paragraph(", ".join(skills))
            for run in skills_para.runs:
                run.font.size = Pt(10)

        doc.save(output_path)
        return f"Resume successfully created at: {output_path}"

    except Exception as e:
        raise Exception(f"Error creating resume: {str(e)}")


@mcp.tool()
def create_cover_letter(
    output_path: str,
    name: str,
    body_paragraphs: list,
    contact: dict = None,
    date: str = None,
    recipient: dict = None,
) -> str:
    """
    Create a professionally formatted cover letter in .docx format
    """
    try:
        doc = Document()
        _setup_page_margins(doc, 1.0)  # Normal margins

        # Applicant's contact info
        name_para = doc.add_paragraph(name)
        name_para.runs[0].font.size = Pt(12)
        name_para.runs[0].font.bold = True
        name_para.paragraph_format.space_after = Pt(0)

        if contact:
            for key in ["address", "phone", "email"]:
                if key in contact:
                    para = doc.add_paragraph(contact[key])
                    para.runs[0].font.size = Pt(11)
                    para.paragraph_format.space_after = Pt(0)
            # Add space after contact block
            doc.add_paragraph().paragraph_format.space_after = Pt(12)

        # Date
        if date:
            date_para = doc.add_paragraph(date)
            date_para.runs[0].font.size = Pt(11)
            date_para.paragraph_format.space_after = Pt(12)

        # Recipient info
        if recipient:
            recipient_lines = []
            seen = set()
            for key in ["name", "title", "company", "address"]:
                if key in recipient and recipient[key]:
                    value = recipient[key].strip()
                    if value.lower() not in seen:
                        recipient_lines.append(value)
                        seen.add(value.lower())

            for i, line in enumerate(recipient_lines):
                rec_para = doc.add_paragraph(line)
                rec_para.runs[0].font.size = Pt(11)
                rec_para.paragraph_format.space_after = (
                    Pt(12) if i == len(recipient_lines) - 1 else Pt(0)
                )

        # Body
        for paragraph in body_paragraphs:
            body_para = doc.add_paragraph(paragraph)
            body_para.paragraph_format.space_after = Pt(12)
            body_para.paragraph_format.line_spacing = 1.15
            for run in body_para.runs:
                run.font.size = Pt(11)

        # Closing
        closing_para = doc.add_paragraph("Sincerely,")
        closing_para.runs[0].font.size = Pt(11)
        closing_para.paragraph_format.space_after = Pt(36)

        signature_para = doc.add_paragraph(name)
        signature_para.runs[0].font.size = Pt(11)

        doc.save(output_path)
        return f"Cover letter successfully created at: {output_path}"

    except Exception as e:
        raise Exception(f"Error creating cover letter: {str(e)}")


@mcp.tool()
def create_formatted_document(
    output_path: str, sections: list, title: str = None
) -> str:
    """
    Create a custom formatted Word document
    """
    try:
        doc = Document()

        if title:
            title_heading = doc.add_heading(title, 0)
            title_heading.alignment = WD_ALIGN_PARAGRAPH.CENTER

        for section in sections:
            if "heading" in section:
                doc.add_heading(section["heading"], 1)

            for item in section.get("content", []):
                item_type = item.get("type", "paragraph")
                text = item.get("text", "")

                if item_type == "paragraph":
                    doc.add_paragraph(text)
                elif item_type == "bullet":
                    doc.add_paragraph(text, style="List Bullet")
                elif item_type == "numbered":
                    doc.add_paragraph(text, style="List Number")

            doc.add_paragraph()

        doc.save(output_path)
        return f"Document successfully created at: {output_path}"

    except Exception as e:
        raise Exception(f"Error creating document: {str(e)}")


if __name__ == "__main__":
    try:
        print("Starting docx-creator MCP server...", file=sys.stderr)
        mcp.run()
    except Exception as e:
        print(f"ERROR: Failed to start server: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
