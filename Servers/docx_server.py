"""MCP Server for creating intelligently formatted Word documents (.docx)."""

import os
import sys
import traceback
from functools import lru_cache
from statistics import mean
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor
from fastmcp import FastMCP

try:
    from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
except ImportError:  # pragma: no cover
    AzureOpenAIEmbeddings = None
    OpenAIEmbeddings = None

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

def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def split_sentences(text: str) -> List[str]:
    sentences: List[str] = []
    current: List[str] = []
    delimiters = {".", "?", "!"}
    for token in text.split():
        current.append(token)
        if token[-1] in delimiters:
            sentences.append(" ".join(current).strip())
            current = []
    if current:
        sentences.append(" ".join(current).strip())
    return [s for s in sentences if len(s.split()) >= 5]


def chunk_text(text: str, max_tokens: int = 120) -> List[str]:
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return [normalize_whitespace(text)]
    chunks: List[str] = []
    for i in range(0, len(tokens), max_tokens):
        chunk = " ".join(tokens[i : i + max_tokens]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def count_numeric_tokens(text: str) -> int:
    return sum(1 for token in text.split() if any(char.isdigit() for char in token))


def compute_readability_score(text: str) -> float:
    sentences = split_sentences(text)
    if not sentences:
        return 0.0
    sentence_lengths = [len(sentence.split()) for sentence in sentences]
    avg_length = mean(sentence_lengths)
    score = max(40.0, min(95.0, 110 - avg_length))
    return round(score, 1)


def compute_story_density(text: str) -> float:
    sentences = split_sentences(text)
    if not sentences:
        return 0.0
    numeric_hits = sum(count_numeric_tokens(sentence) > 0 for sentence in sentences)
    density = (numeric_hits / len(sentences)) * 100
    return round(min(100.0, max(20.0, density)), 1)


def detect_sections(text: str) -> Dict[str, str]:
    sections: Dict[str, List[str]] = {}
    current_key = "General"
    sections[current_key] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.isupper() and len(stripped.split()) <= 4:
            current_key = stripped.title()
            sections.setdefault(current_key, [])
        else:
            sections[current_key].append(stripped)
    return {key: " ".join(value) for key, value in sections.items() if value}


@lru_cache(maxsize=1)
def get_embedding_model():
    azure_vars = {
        "key": os.getenv("AZURE_OPENAI_API_KEY"),
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "version": os.getenv("AZURE_OPENAI_API_VERSION"),
        "deployment": os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
    }
    if all(azure_vars.values()) and AzureOpenAIEmbeddings is not None:
        return AzureOpenAIEmbeddings(
            api_key=azure_vars["key"],
            azure_endpoint=azure_vars["endpoint"],
            api_version=azure_vars["version"],
            deployment=azure_vars["deployment"],
        )

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and OpenAIEmbeddings is not None:
        return OpenAIEmbeddings(api_key=openai_key)

    return None


def embed_text_blocks(texts: List[str]) -> Optional[np.ndarray]:
    model = get_embedding_model()
    if not model or not texts:
        return None
    vectors = model.embed_documents(texts)
    return np.array(vectors, dtype=np.float32)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)))
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-8)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-8)
    return a_norm @ b_norm.T


def summarize_alignment(resume_chunks: List[str], job_chunks: List[str]) -> Dict[str, List[Tuple[str, float, str]]]:
    resume_vecs = embed_text_blocks(resume_chunks)
    job_vecs = embed_text_blocks(job_chunks)
    if resume_vecs is None or job_vecs is None:
        return {"highlights": [], "gaps": []}

    sim_matrix = cosine_similarity_matrix(resume_vecs, job_vecs)
    highlights: List[Tuple[str, float, str]] = []
    gaps: List[Tuple[str, float, str]] = []

    if sim_matrix.size:
        resume_best_idx = np.argmax(sim_matrix, axis=1)
        resume_best_scores = sim_matrix[np.arange(len(resume_chunks)), resume_best_idx]
        top_indices = np.argsort(resume_best_scores)[::-1][:5]
        for idx in top_indices:
            highlights.append(
                (
                    resume_chunks[idx],
                    float(round(resume_best_scores[idx], 3)),
                    job_chunks[resume_best_idx[idx]],
                )
            )

        job_best_idx = np.argmax(sim_matrix, axis=0)
        job_best_scores = sim_matrix[job_best_idx, np.arange(len(job_chunks))]
        low_indices = np.argsort(job_best_scores)[:5]
        for idx in low_indices:
            gaps.append(
                (
                    job_chunks[idx],
                    float(round(job_best_scores[idx], 3)),
                    resume_chunks[job_best_idx[idx]] if len(resume_chunks) else "",
                )
            )

    return {"highlights": highlights, "gaps": gaps}


def compute_skill_scores(resume_text: str, job_text: str) -> Tuple[float, float, float, List[str], List[str]]:
    keywords = {
        "technical": [
            "python",
            "java",
            "c++",
            "javascript",
            "typescript",
            "sql",
            "nosql",
            "react",
            "angular",
            "vue",
            "aws",
            "azure",
            "gcp",
            "docker",
            "kubernetes",
            "terraform",
            "ansible",
            "jenkins",
            "ci/cd",
            "machine learning",
            "deep learning",
            "nlp",
            "data analysis",
            "etl",
            "product management",
            "ui/ux",
            "design",
            "graphql",
            "microservices",
            "serverless",
        ],
        "soft": [
            "leadership",
            "communication",
            "collaboration",
            "stakeholder",
            "mentoring",
            "strategic",
            "analytical",
            "problem solving",
            "negotiation",
            "presentation",
            "cross-functional",
            "agile",
            "scrum",
        ],
    }

    resume_lower = resume_text.lower()
    job_lower = job_text.lower()

    matched: List[str] = []
    missing: List[str] = []
    for domain_keywords in keywords.values():
        for keyword in domain_keywords:
            if keyword in job_lower:
                if keyword in resume_lower:
                    matched.append(keyword)
                else:
                    missing.append(keyword)

    total = len(matched) + len(missing)
    coverage = (len(matched) / total) * 100 if total else 0
    story = compute_story_density(resume_text)
    readability = compute_readability_score(resume_text)
    return round(coverage, 1), round(story, 1), round(readability, 1), matched, missing


def recommendation_from_missing(missing_keywords: Iterable[str]) -> List[str]:
    recs: List[str] = []
    showcased = sorted(set(missing_keywords))[:6]
    for keyword in showcased:
        recs.append(
            f"Weave in a quantified example showcasing **{keyword.title()}**—consider framing it as impact, scope, and tools used."
        )
    return recs


def build_score_table(doc: Document, scores: Dict[str, Tuple[float, str]]):
    table = doc.add_table(rows=1, cols=3)
    table.style = "Light Shading Accent 1"
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = "Dimension"
    hdr_cells[1].text = "Score"
    hdr_cells[2].text = "What it means"
    for cell in hdr_cells:
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.font.bold = True
    for dimension, (score, context) in scores.items():
        row_cells = table.add_row().cells
        row_cells[0].text = dimension
        row_cells[1].text = f"{score:.0f}/100"
        row_cells[2].text = context


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

@mcp.tool()
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

        sections = doc.sections
        for section in sections:
            section.top_margin = Inches(0.75)
            section.bottom_margin = Inches(0.75)
            section.left_margin = Inches(0.9)
            section.right_margin = Inches(0.9)

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

        coverage_score, story_score, readability_score, matched_keywords, missing_keywords = compute_skill_scores(
            resume_text, job_description_text
        )
        total_score = round((coverage_score * 0.5) + (story_score * 0.3) + (readability_score * 0.2), 1)

        scores = {
            "Overall Alignment": (total_score, "Composite of skill coverage, achievements, and readability."),
            "Skill Coverage": (coverage_score, "Portion of job-required keywords already present."),
            "Impact Density": (story_score, "Frequency of quantified achievements in your stories."),
            "Readability": (readability_score, "How easy the resume reads for hiring managers and ATS."),
        }

        add_section_heading(doc, "Executive Summary")
        doc.add_paragraph(
            "We scored this resume against the target role's expectations. Use the insights below to fine-tune "
            "language, highlight peak impact, and close requirement gaps."
        )
        build_score_table(doc, scores)

        resume_sections = detect_sections(resume_text)
        job_sections = detect_sections(job_description_text)
        resume_chunks = chunk_text(resume_text, max_tokens=100)
        job_chunks = chunk_text(job_description_text, max_tokens=80)
        alignment = summarize_alignment(resume_chunks, job_chunks)

        add_section_heading(doc, "High-Confidence Strengths")
        if alignment["highlights"]:
            for resume_snippet, score, job_snippet in alignment["highlights"]:
                quote = doc.add_paragraph(style="Intense Quote")
                quote.add_run(f"Resume Highlight (similarity {score:.2f}):\n").bold = True
                quote.add_run(normalize_whitespace(resume_snippet))
                quote.add_run("\nJob Hook: ").bold = True
                quote.add_run(normalize_whitespace(job_snippet))
        elif matched_keywords:
            doc.add_paragraph(
                "Resume messaging aligns strongly with the role on the following skills:" + 
                ", ".join(sorted({kw.title() for kw in matched_keywords})),
                style="Intense Quote"
            )
        else:
            doc.add_paragraph(
                "Limited overlap detected. Anchor future edits around leadership achievements and quantifiable impact.",
                style="Intense Quote",
            )

        add_section_heading(doc, "Gap Radar")
        if alignment["gaps"]:
            for job_snippet, score, resume_snippet in alignment["gaps"]:
                para = doc.add_paragraph()
                para.add_run("Job Priority: ").bold = True
                para.add_run(normalize_whitespace(job_snippet))
                para.add_run(f" (coverage score {score:.2f})\n")
                if resume_snippet:
                    para.add_run("Closest Resume Signal: ").bold = True
                    para.add_run(normalize_whitespace(resume_snippet))
                else:
                    para.add_run("No corresponding resume content detected.")
        else:
            doc.add_paragraph(
                "Great coverage! We could not find any major gaps between the job description and your resume messaging.",
                style="Intense Quote",
            )

        add_section_heading(doc, "Keyword Opportunities")
        if missing_keywords:
            doc.add_paragraph(
                "Introduce or amplify the following capabilities to mirror the job description:", style="Intense Quote"
            )
            for keyword in sorted(set(missing_keywords)):
                doc.add_paragraph(keyword.title(), style="List Bullet").runs[0].font.size = Pt(10)
        else:
            doc.add_paragraph("Resume already reflects the critical keywords present in the job description.", style="Intense Quote")

        add_section_heading(doc, "Tailored Recommendations")
        recs = recommendation_from_missing(missing_keywords)
        if not recs:
            recs = [
                "Maintain quantified achievements at the top of each experience section.",
                "Align section ordering so the most relevant projects appear first.",
            ]
        for rec in recs:
            doc.add_paragraph(rec, style="List Number")

        add_section_heading(doc, "Role Readiness Checklist")
        checklist_items = [
            "Professional summary mirrors the job title and embeds 2–3 key differentiators.",
            "Each recent role has at least one measurable outcome (%, $, time saved).",
            "Keywords from the job description are placed in both bullets and skills.",
            "Resume length stays within 1–2 pages with whitespace preserved for readability.",
        ]
        for item in checklist_items:
            doc.add_paragraph(item, style="List Bullet")

        doc.save(output_path)
        return f"Resume analysis report successfully created at: {output_path}"

    except Exception as exc:  # pragma: no cover
        raise Exception(f"Error creating resume analysis report: {exc}")

if __name__ == "__main__":
    try:
        print("Starting docx-creator MCP server...", file=sys.stderr)
        mcp.run()
    except Exception as e:
        print(f"ERROR: Failed to start server: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)