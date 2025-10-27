#!/usr/bin/env python3
"""
MCP Server for creating properly formatted Word documents (.docx)
"""

import asyncio
import json
from typing import Any
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio

# Initialize MCP server
app = Server("docx-creator")

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools for document creation"""
    return [
        Tool(
            name="create_resume",
            description="Create a professionally formatted resume in .docx format",
            inputSchema={
                "type": "object",
                "properties": {
                    "output_path": {
                        "type": "string",
                        "description": "Full path where the .docx file should be saved"
                    },
                    "name": {
                        "type": "string",
                        "description": "Candidate's full name"
                    },
                    "contact": {
                        "type": "object",
                        "description": "Contact information",
                        "properties": {
                            "email": {"type": "string"},
                            "phone": {"type": "string"},
                            "location": {"type": "string"},
                            "linkedin": {"type": "string"}
                        }
                    },
                    "summary": {
                        "type": "string",
                        "description": "Professional summary or objective"
                    },
                    "experience": {
                        "type": "array",
                        "description": "Work experience entries",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "company": {"type": "string"},
                                "location": {"type": "string"},
                                "dates": {"type": "string"},
                                "responsibilities": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            }
                        }
                    },
                    "education": {
                        "type": "array",
                        "description": "Education entries",
                        "items": {
                            "type": "object",
                            "properties": {
                                "degree": {"type": "string"},
                                "school": {"type": "string"},
                                "location": {"type": "string"},
                                "graduation": {"type": "string"}
                            }
                        }
                    },
                    "skills": {
                        "type": "array",
                        "description": "List of skills",
                        "items": {"type": "string"}
                    }
                },
                "required": ["output_path", "name"]
            }
        ),
        Tool(
            name="create_cover_letter",
            description="Create a professionally formatted cover letter in .docx format",
            inputSchema={
                "type": "object",
                "properties": {
                    "output_path": {
                        "type": "string",
                        "description": "Full path where the .docx file should be saved"
                    },
                    "name": {
                        "type": "string",
                        "description": "Applicant's full name"
                    },
                    "contact": {
                        "type": "object",
                        "properties": {
                            "email": {"type": "string"},
                            "phone": {"type": "string"},
                            "address": {"type": "string"}
                        }
                    },
                    "date": {
                        "type": "string",
                        "description": "Date of letter"
                    },
                    "recipient": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "title": {"type": "string"},
                            "company": {"type": "string"},
                            "address": {"type": "string"}
                        }
                    },
                    "body_paragraphs": {
                        "type": "array",
                        "description": "Paragraphs of the cover letter body",
                        "items": {"type": "string"}
                    }
                },
                "required": ["output_path", "name", "body_paragraphs"]
            }
        ),
        Tool(
            name="create_formatted_document",
            description="Create a custom formatted Word document with headings, paragraphs, and lists",
            inputSchema={
                "type": "object",
                "properties": {
                    "output_path": {
                        "type": "string",
                        "description": "Full path where the .docx file should be saved"
                    },
                    "title": {
                        "type": "string",
                        "description": "Document title"
                    },
                    "sections": {
                        "type": "array",
                        "description": "Sections of the document",
                        "items": {
                            "type": "object",
                            "properties": {
                                "heading": {"type": "string"},
                                "content": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "type": {
                                                "type": "string",
                                                "enum": ["paragraph", "bullet", "numbered"]
                                            },
                                            "text": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "required": ["output_path", "sections"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls"""
    
    if name == "create_resume":
        return await create_resume(arguments)
    elif name == "create_cover_letter":
        return await create_cover_letter(arguments)
    elif name == "create_formatted_document":
        return await create_formatted_document(arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")

async def create_resume(args: dict) -> list[TextContent]:
    """Create a professionally formatted resume"""
    try:
        doc = Document()
        
        # Set margins
        sections = doc.sections
        for section in sections:
            section.top_margin = Inches(0.5)
            section.bottom_margin = Inches(0.5)
            section.left_margin = Inches(0.75)
            section.right_margin = Inches(0.75)
        
        # Name (centered, large)
        name_para = doc.add_paragraph(args['name'])
        name_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        name_run = name_para.runs[0]
        name_run.font.size = Pt(20)
        name_run.font.bold = True
        
        # Contact info (centered)
        if 'contact' in args:
            contact = args['contact']
            contact_parts = []
            if 'email' in contact:
                contact_parts.append(contact['email'])
            if 'phone' in contact:
                contact_parts.append(contact['phone'])
            if 'location' in contact:
                contact_parts.append(contact['location'])
            if 'linkedin' in contact:
                contact_parts.append(contact['linkedin'])
            
            contact_para = doc.add_paragraph(' | '.join(contact_parts))
            contact_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            contact_para.runs[0].font.size = Pt(10)
        
        doc.add_paragraph()  # Space
        
        # Summary
        if 'summary' in args:
            add_section_heading(doc, "PROFESSIONAL SUMMARY")
            doc.add_paragraph(args['summary'])
            doc.add_paragraph()  # Space
        
        # Experience
        if 'experience' in args and args['experience']:
            add_section_heading(doc, "EXPERIENCE")
            for exp in args['experience']:
                # Job title and company (bold)
                title_para = doc.add_paragraph()
                title_run = title_para.add_run(f"{exp.get('title', '')} - {exp.get('company', '')}")
                title_run.bold = True
                title_run.font.size = Pt(11)
                
                # Location and dates (italic)
                details_para = doc.add_paragraph()
                details_run = details_para.add_run(
                    f"{exp.get('location', '')} | {exp.get('dates', '')}"
                )
                details_run.italic = True
                details_run.font.size = Pt(10)
                
                # Responsibilities (bullets)
                if 'responsibilities' in exp:
                    for resp in exp['responsibilities']:
                        doc.add_paragraph(resp, style='List Bullet')
                
                doc.add_paragraph()  # Space between jobs
        
        # Education
        if 'education' in args and args['education']:
            add_section_heading(doc, "EDUCATION")
            for edu in args['education']:
                edu_para = doc.add_paragraph()
                degree_run = edu_para.add_run(edu.get('degree', ''))
                degree_run.bold = True
                
                school_para = doc.add_paragraph(
                    f"{edu.get('school', '')} - {edu.get('location', '')} | {edu.get('graduation', '')}"
                )
                school_para.runs[0].font.size = Pt(10)
            
            doc.add_paragraph()  # Space
        
        # Skills
        if 'skills' in args and args['skills']:
            add_section_heading(doc, "SKILLS")
            doc.add_paragraph(', '.join(args['skills']))
        
        # Save document
        doc.save(args['output_path'])
        
        return [TextContent(
            type="text",
            text=f"Resume successfully created at: {args['output_path']}"
        )]
        
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Error creating resume: {str(e)}"
        )]

async def create_cover_letter(args: dict) -> list[TextContent]:
    """Create a professionally formatted cover letter"""
    try:
        doc = Document()
        
        # Set margins
        sections = doc.sections
        for section in sections:
            section.top_margin = Inches(1)
            section.bottom_margin = Inches(1)
            section.left_margin = Inches(1)
            section.right_margin = Inches(1)
        
        # Applicant's contact info
        doc.add_paragraph(args['name'])
        if 'contact' in args:
            contact = args['contact']
            if 'address' in contact:
                doc.add_paragraph(contact['address'])
            if 'phone' in contact:
                doc.add_paragraph(contact['phone'])
            if 'email' in contact:
                doc.add_paragraph(contact['email'])
        
        doc.add_paragraph()  # Space
        
        # Date
        if 'date' in args:
            doc.add_paragraph(args['date'])
        
        doc.add_paragraph()  # Space
        
        # Recipient info
        if 'recipient' in args:
            recipient = args['recipient']
            if 'name' in recipient:
                doc.add_paragraph(recipient['name'])
            if 'title' in recipient:
                doc.add_paragraph(recipient['title'])
            if 'company' in recipient:
                doc.add_paragraph(recipient['company'])
            if 'address' in recipient:
                doc.add_paragraph(recipient['address'])
        
        doc.add_paragraph()  # Space
        
        # Salutation
        salutation = "Dear Hiring Manager,"
        if 'recipient' in args and 'name' in args['recipient']:
            salutation = f"Dear {args['recipient']['name']},"
        doc.add_paragraph(salutation)
        
        doc.add_paragraph()  # Space
        
        # Body paragraphs
        for paragraph in args.get('body_paragraphs', []):
            doc.add_paragraph(paragraph)
            doc.add_paragraph()  # Space between paragraphs
        
        # Closing
        doc.add_paragraph("Sincerely,")
        doc.add_paragraph()
        doc.add_paragraph(args['name'])
        
        # Save document
        doc.save(args['output_path'])
        
        return [TextContent(
            type="text",
            text=f"Cover letter successfully created at: {args['output_path']}"
        )]
        
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Error creating cover letter: {str(e)}"
        )]

async def create_formatted_document(args: dict) -> list[TextContent]:
    """Create a custom formatted document"""
    try:
        doc = Document()
        
        # Add title if provided
        if 'title' in args:
            title = doc.add_heading(args['title'], 0)
            title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Add sections
        for section in args.get('sections', []):
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
        doc.save(args['output_path'])
        
        return [TextContent(
            type="text",
            text=f"Document successfully created at: {args['output_path']}"
        )]
        
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Error creating document: {str(e)}"
        )]

def add_section_heading(doc: Document, text: str):
    """Add a formatted section heading"""
    heading = doc.add_paragraph(text)
    heading_run = heading.runs[0]
    heading_run.font.size = Pt(12)
    heading_run.font.bold = True
    heading_run.font.color.rgb = RGBColor(0, 0, 0)
    
    # Add a bottom border for the heading
    heading.paragraph_format.space_after = Pt(6)

async def main():
    """Run the MCP server"""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())