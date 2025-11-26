"""
MCP Server for creating properly formatted Word documents (.docx)
Returns documents as base64 encoded strings for Slack integration
"""

import sys
import traceback
import base64
import io
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
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

@mcp.tool()
def create_resume(
    name: str,
    filename: str = "resume.docx",
    contact: dict = None,
    summary: str = None,
    experience: list = None,
    education: list = None,
    skills: list = None
) -> dict:
    """
    Create a professionally formatted resume in .docx format (optimized for one page)
    Returns the document as base64 encoded string for delivery via Slack
    
    Args:
        name: Candidate's full name
        filename: Name for the document file
        contact: Contact information (email, phone, location, linkedin)
        summary: Professional summary or objective
        experience: List of work experience entries with title, company, location, dates, responsibilities
        education: List of education entries with degree, school, location, graduation
        skills: List of skills
    
    Returns:
        Dictionary with filename and base64 encoded document content
    """
    try:
        doc = Document()
        
        # Set tight margins for one-page layout
        sections = doc.sections
        for section in sections:
            section.top_margin = Inches(0.4)
            section.bottom_margin = Inches(0.4)
            section.left_margin = Inches(0.5)
            section.right_margin = Inches(0.5)
        
        # Name (centered, slightly smaller for space efficiency)
        name_para = doc.add_paragraph(name)
        name_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        name_run = name_para.runs[0]
        name_run.font.size = Pt(16)
        name_run.font.bold = True
        name_para.paragraph_format.space_after = Pt(2)
        
        # Contact info (centered, compact)
        if contact:
            contact_parts = []
            if 'email' in contact:
                contact_parts.append(contact['email'])
            if 'phone' in contact:
                contact_parts.append(contact['phone'])
            if 'location' in contact:
                contact_parts.append(contact['location'])
            if 'linkedin' in contact:
                contact_parts.append(contact['linkedin'])
            
            if contact_parts:
                contact_para = doc.add_paragraph(' | '.join(contact_parts))
                contact_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                contact_para.runs[0].font.size = Pt(9)
                contact_para.paragraph_format.space_after = Pt(6)
        
        # Summary
        if summary:
            add_section_heading(doc, "PROFESSIONAL SUMMARY")
            summary_para = doc.add_paragraph(summary)
            summary_para.paragraph_format.space_after = Pt(6)
            for run in summary_para.runs:
                run.font.size = Pt(10)
        
        # Education
        if education:
            add_section_heading(doc, "EDUCATION")
            for edu in education:
                edu_para = doc.add_paragraph()
                edu_para.paragraph_format.space_after = Pt(1)
                degree_run = edu_para.add_run(edu.get('degree', ''))
                degree_run.bold = True
                degree_run.font.size = Pt(10)
                
                school_para = doc.add_paragraph(
                    f"{edu.get('school', '')} - {edu.get('location', '')} | {edu.get('graduation', '')}"
                )
                school_para.paragraph_format.space_after = Pt(4)
                school_para.runs[0].font.size = Pt(9)
        
        # Experience
        if experience:
            add_section_heading(doc, "EXPERIENCE")
            for i, exp in enumerate(experience):
                # Job title and company (bold)
                title_para = doc.add_paragraph()
                title_para.paragraph_format.space_after = Pt(1)
                title_run = title_para.add_run(f"{exp.get('title', '')} - {exp.get('company', '')}")
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
                if 'responsibilities' in exp:
                    for resp in exp['responsibilities']:
                        bullet_para = doc.add_paragraph(resp, style='List Bullet')
                        bullet_para.paragraph_format.space_after = Pt(1)
                        bullet_para.paragraph_format.line_spacing = 1.0
                        for run in bullet_para.runs:
                            run.font.size = Pt(10)
                
                # Space between jobs (smaller)
                if i < len(experience) - 1:
                    space_para = doc.add_paragraph()
                    space_para.paragraph_format.space_after = Pt(4)
        
        # Skills
        if skills:
            add_section_heading(doc, "SKILLS")
            skills_para = doc.add_paragraph(', '.join(skills))
            for run in skills_para.runs:
                run.font.size = Pt(10)
        
        # Save document to memory buffer
        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        
        # Encode as base64
        doc_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return {
            "filename": filename,
            "content": doc_base64,
            "message": f"Resume successfully created: {filename}"
        }
        
    except Exception as e:
        raise Exception(f"Error creating resume: {str(e)}")

@mcp.tool()
def create_cover_letter(
    name: str,
    body_paragraphs: list,
    filename: str = "cover_letter.docx",
    contact: dict = None,
    date: str = None,
    recipient: dict = None
) -> dict:
    """
    Create a professionally formatted cover letter in .docx format (optimized for one page)
    Returns the document as base64 encoded string for delivery via Slack
    
    Args:
        name: Applicant's full name
        body_paragraphs: List of paragraphs for the cover letter body
        filename: Name for the document file
        contact: Contact information (email, phone, address)
        date: Date of letter
        recipient: Recipient information (name, title, company, address)
    
    Returns:
        Dictionary with filename and base64 encoded document content
    """
    try:
        doc = Document()
        
        # Set normal margins
        sections = doc.sections
        for section in sections:
            section.top_margin = Inches(1)
            section.bottom_margin = Inches(1)
            section.left_margin = Inches(1)
            section.right_margin = Inches(1)
        
        # Applicant's contact info (standard size)
        name_para = doc.add_paragraph(name)
        name_para.runs[0].font.size = Pt(12)
        name_para.runs[0].font.bold = True
        name_para.paragraph_format.space_after = Pt(0)
        
        if contact:
            if 'address' in contact:
                addr_para = doc.add_paragraph(contact['address'])
                addr_para.runs[0].font.size = Pt(11)
                addr_para.paragraph_format.space_after = Pt(0)
            if 'phone' in contact:
                phone_para = doc.add_paragraph(contact['phone'])
                phone_para.runs[0].font.size = Pt(11)
                phone_para.paragraph_format.space_after = Pt(0)
            if 'email' in contact:
                email_para = doc.add_paragraph(contact['email'])
                email_para.runs[0].font.size = Pt(11)
                email_para.paragraph_format.space_after = Pt(12)
        
        # Date
        if date:
            date_para = doc.add_paragraph(date)
            date_para.runs[0].font.size = Pt(11)
            date_para.paragraph_format.space_after = Pt(12)
        
        # Recipient info (with deduplication)
        if recipient:
            recipient_lines = []
            seen = set()
            
            for key in ['name', 'title', 'company', 'address']:
                if key in recipient and recipient[key]:
                    value = recipient[key].strip()
                    value_lower = value.lower()
                    # Skip if we've already added this exact text
                    if value_lower not in seen:
                        recipient_lines.append(value)
                        seen.add(value_lower)
            
            for i, line in enumerate(recipient_lines):
                rec_para = doc.add_paragraph(line)
                rec_para.runs[0].font.size = Pt(11)
                # Add spacing after last line
                if i == len(recipient_lines) - 1:
                    rec_para.paragraph_format.space_after = Pt(12)
                else:
                    rec_para.paragraph_format.space_after = Pt(0)
        
        # Body paragraphs (normal spacing, 1.15 line spacing)
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
        
        # Save document to memory buffer
        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        
        # Encode as base64
        doc_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return {
            "filename": filename,
            "content": doc_base64,
            "message": f"Cover letter successfully created: {filename}"
        }
        
    except Exception as e:
        raise Exception(f"Error creating cover letter: {str(e)}")

@mcp.tool()
def create_formatted_document(
    sections: list,
    filename: str = "document.docx",
    title: str = None
) -> dict:
    """
    Create a custom formatted Word document with headings, paragraphs, and lists
    Returns the document as base64 encoded string for delivery via Slack
    
    Args:
        sections: List of sections with heading and content (paragraphs, bullets, numbered lists)
        filename: Name for the document file
        title: Optional document title
    
    Returns:
        Dictionary with filename and base64 encoded document content
    """
    try:
        doc = Document()
        
        # Add title if provided
        if title:
            title_heading = doc.add_heading(title, 0)
            title_heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Add sections
        for section in sections:
            if 'heading' in section:
                doc.add_heading(section['heading'], 1)
            
            for item in section.get('content', []):
                item_type = item.get('type', 'paragraph')
                text = item.get('text', '')
                
                if item_type == 'paragraph':
                    doc.add_paragraph(text)
                elif item_type == 'bullet':
                    doc.add_paragraph(text, style='List Bullet')
                elif item_type == 'numbered':
                    doc.add_paragraph(text, style='List Number')
            
            doc.add_paragraph()  # Space between sections
        
        # Save document to memory buffer
        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        
        # Encode as base64
        doc_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return {
            "filename": filename,
            "content": doc_base64,
            "message": f"Document successfully created: {filename}"
        }
        
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