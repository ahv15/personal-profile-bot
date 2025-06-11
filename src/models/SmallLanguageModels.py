#!/usr/bin/env python
# coding: utf-8

"""
Small Language Models module for personal profile bot.

This module provides functionality for extracting text from PDF resumes
and generating question-answer pairs using OpenAI's language models.
"""

import json
import time
from typing import List, Dict, Any, Optional

from PyPDF2 import PdfReader
from pydantic import BaseModel
from openai import OpenAI


class QuestionAnswerPair(BaseModel):
    """Pydantic model for validating question-answer pair responses from the API."""
    question: List[str]
    answer: List[str]


def extract_resume_text(pdf_path: str) -> str:
    """
    Extract text content from a PDF resume.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text content from all pages
        
    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        Exception: If PDF reading fails
    """
    reader = PdfReader(pdf_path)
    pages = []
    
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
            
    return "\n\n".join(pages).strip()


def generate_unique_qa_pairs(
    client: OpenAI,
    resume_text: str,
    target_count: int = 2000,
    batch_size: int = 100,
    model: str = "gpt-4o",
    pause_s: float = 1.0
) -> List[Dict[str, str]]:
    """
    Generate unique question-answer pairs from resume text.
    
    This function generates QA pairs in batches, filtering out duplicates
    to ensure uniqueness across the entire dataset.
    
    Args:
        client (OpenAI): OpenAI client instance
        resume_text (str): The resume text to generate QA pairs from
        target_count (int, optional): Target number of unique QA pairs. Defaults to 2000.
        batch_size (int, optional): Number of pairs to generate per API call. Defaults to 100.
        model (str, optional): OpenAI model to use. Defaults to "gpt-4o".
        pause_s (float, optional): Pause between API calls in seconds. Defaults to 1.0.
        
    Returns:
        List[Dict[str, str]]: List of dictionaries with "question" and "answer" keys
        
    Raises:
        Exception: If API calls fail or response parsing errors occur
    """
    seen_questions = set()
    results = []
    
    while len(results) < target_count:
        # Build the prompt for the current batch
        prompt = (
            "You are a helpful assistant. Given the following full resume text, "
            f"generate {batch_size} useful question-answer pairs that a visitor might ask about the candidate. "
            "Return only a JSON array of objects with \"question\" and \"answer\" fields.\n\n"
            "Full resume text:\n\"\"\"\n"
            + resume_text +
            "\n\"\"\"\n"
        )
        
        # Make API call with structured output
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "system", 
                    "content": "You generate concise QA pairs from resume text."
                },
                {
                    "role": "user", 
                    "content": prompt
                },
            ],
            response_format=QuestionAnswerPair,
            temperature=0.1
        )
        
        # Parse response and extract QA pairs
        event = response.choices[0].message.parsed
        questions = event.question
        answers = event.answer
        new_this_round = 0
        
        # Process each question-answer pair
        for i, question in enumerate(questions):
            if question not in seen_questions:
                seen_questions.add(question)
                results.append({
                    "question": question,
                    "answer": answers[i]
                })
                new_this_round += 1
        
        # Progress reporting
        num_fetched = len(questions)
        print(
            f"Batch fetched: {num_fetched} pairs → "
            f"{new_this_round} new → {len(results)}/{target_count}"
        )
        
        # Check for early termination if no new unique questions
        if new_this_round == 0:
            print("No new unique questions—stopping early to avoid infinite loop.")
            break
        
        # Rate limiting
        time.sleep(pause_s)
    
    return results[:target_count]
