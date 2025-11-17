"""
MCP Server for creating properly formatted Word documents (.docx)
"""

import sys
import traceback
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
    output_path: str,
    name: str,
    contact: dict = None,
    summary: str = None,
    experience: list = None,
    education: list = None,
    skills: list = None
) -> str:
    """
    Create a professionally formatted resume in .docx format (optimized for one page)
    
    Args:
        output_path: Full path where the .docx file should be saved
        name: Candidate's full name
        contact: Contact information (email, phone, location, linkedin)
        summary: Professional summary or objective
        experience: List of work experience entries with title, company, location, dates, responsibilities
        education: List of education entries with degree, school, location, graduation
        skills: List of skills
    
    Returns:
        Success message with file path
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
        
        # Save document
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
    recipient: dict = None
) -> str:
    """
    Create a professionally formatted cover letter in .docx format (optimized for one page)
    
    Args:
        output_path: Full path where the .docx file should be saved
        name: Applicant's full name
        body_paragraphs: List of paragraphs for the cover letter body
        contact: Contact information (email, phone, address)
        date: Date of letter
        recipient: Recipient information (name, title, company, address)
    
    Returns:
        Success message with file path
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
        
        # Save document
        doc.save(output_path)
        
        return f"Cover letter successfully created at: {output_path}"
        
    except Exception as e:
        raise Exception(f"Error creating cover letter: {str(e)}")

@mcp.tool()
def create_formatted_document(
    output_path: str,
    sections: list,
    title: str = None
) -> str:
    """
    Create a custom formatted Word document with headings, paragraphs, and lists
    
    Args:
        output_path: Full path where the .docx file should be saved
        sections: List of sections with heading and content (paragraphs, bullets, numbered lists)
        title: Optional document title
    
    Returns:
        Success message with file path
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
        
        # Save document
        doc.save(output_path)
        
        return f"Document successfully created at: {output_path}"
        
    except Exception as e:
        raise Exception(f"Error creating document: {str(e)}")

def create_resume_analysis_report(
    output_path: str,
    resume_text: str,
    job_description_text: str,
    candidate_name: str = "Candidate"
) -> str:
    """
    Analyzes a resume against a job description and creates a .docx feedback report.
    This tool provides "sentient" analysis by comparing documents and offering strategic advice.
    
    Args:
        output_path: Full path where the .docx feedback report should be saved.
        resume_text: The full text of the candidate's resume.
        job_description_text: The full text of the target job description.
        candidate_name: The candidate's name for personalizing the report.
        
    Returns:
        Success message with the file path of the generated report.
    """
    try:
        doc = Document()
        
        # --- Document Setup ---
        sections = doc.sections
        for section in sections:
            section.top_margin = Inches(0.75)
            section.bottom_margin = Inches(0.75)
            section.left_margin = Inches(1)
            section.right_margin = Inches(1)

        # --- Report Header ---
        title_para = doc.add_paragraph("Resume Analysis & Tailoring Report")
        title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        title_run = title_para.runs[0]
        title_run.font.size = Pt(18)
        title_run.font.bold = True
        
        subtitle_para = doc.add_paragraph(f"Prepared for {candidate_name}")
        subtitle_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        subtitle_para.runs[0].font.size = Pt(12)
        subtitle_para.runs[0].italic = True
        subtitle_para.paragraph_format.space_after = Pt(18)

        # --- Simple Keyword Analysis ---
        # A more "sentient" version would use NLP, but this is a good start.
        common_skills = [
            'python', 'java', 'c++', 'javascript', 'typescript', 'sql', 'nosql', 'react', 'angular', 'vue',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'ansible', 'jenkins', 'ci/cd',
            'machine learning', 'deep learning', 'nlp', 'computer vision', 'data analysis', 'etl',
            'project management', 'agile', 'scrum', 'product management', 'ui/ux', 'design'
        ]
        
        jd_lower = job_description_text.lower()
        resume_lower = resume_text.lower()
        
        jd_keywords = {skill for skill in common_skills if skill in jd_lower}
        resume_keywords = {skill for skill in common_skills if skill in resume_lower}
        
        matched_keywords = jd_keywords.intersection(resume_keywords)
        missing_keywords = jd_keywords.difference(resume_keywords)

        # --- Report Content ---
        add_section_heading(doc, "Analysis Summary")
        doc.add_paragraph(
            "This report provides feedback on how well your resume aligns with the target job description. "
            "Use these suggestions to tailor your application and increase your chances of success."
        ).runs[0].font.size = Pt(11)

        # Matched Keywords
        add_section_heading(doc, "Strengths: Matched Keywords")
        if matched_keywords:
            doc.add_paragraph(
                "Your resume effectively highlights the following skills mentioned in the job description:",
                style='Intense Quote'
            )
            for skill in sorted(list(matched_keywords)):
                doc.add_paragraph(skill.title(), style='List Bullet').runs[0].font.size = Pt(10)
        else:
            doc.add_paragraph(
                "No significant keyword overlap was found. It is highly recommended to tailor your resume.",
                style='Intense Quote'
            )
        doc.paragraphs[-1].paragraph_format.space_after = Pt(12)

        # Missing Keywords
        add_section_heading(doc, "Opportunities: Missing Keywords")
        if missing_keywords:
            doc.add_paragraph(
                "Consider incorporating the following keywords from the job description if you have experience with them:",
                style='Intense Quote'
            )
            for skill in sorted(list(missing_keywords)):
                doc.add_paragraph(skill.title(), style='List Bullet').runs[0].font.size = Pt(10)
        else:
            doc.add_paragraph(
                "Great job! Your resume appears to contain all the key skills identified in the job description.",
                style='Intense Quote'
            )
        doc.paragraphs[-1].paragraph_format.space_after = Pt(12)
        
        # Actionable Advice
        add_section_heading(doc, "Recommendations")
        doc.add_paragraph(
            "1. Quantify Achievements: Where possible, use numbers to describe your impact (e.g., 'Increased efficiency by 20%' or 'Managed a budget of $50k').",
            style='List Number'
        )
        doc.add_paragraph(
            "2. Mirror Language: Use similar phrasing and terminology as the job description to pass through automated screening systems (ATS).",
            style='List Number'
        )
        doc.add_paragraph(
            "3. Review and Refine: Ensure your professional summary and recent experience directly address the top requirements listed in the job posting.",
            style='List Number'
        )

        # --- Save Document ---
        doc.save(output_path)
        
        return f"Resume analysis report successfully created at: {output_path}"

    except Exception as e:
        raise Exception(f"Error creating resume analysis report: {str(e)}")

if __name__ == "__main__":
    try:
        print("Starting docx-creator MCP server...", file=sys.stderr)
        mcp.run()
    except Exception as e:
        print(f"ERROR: Failed to start server: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)