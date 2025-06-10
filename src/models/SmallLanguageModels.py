#!/usr/bin/env python
# coding: utf-8

import os
import json
import time
import openai
from PyPDF2 import PdfReader
from pydantic import BaseModel
from openai import OpenAI


class QuestionAnswerPair(BaseModel):
    question: list[str]
    answer: list[str]


def extract_resume_text(pdf_path: str) -> str:
    """Extract text content from a PDF resume."""
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages).strip()


def generate_unique_qa_pairs(
    client,
    resume_text: str,
    target_count: int = 2000,
    batch_size: int = 100,
    model: str = "gpt-4o",
    pause_s: float = 1.0
):
    """Generate unique question-answer pairs from resume text."""
    seen_questions = set()
    results = []

    while len(results) < target_count:
        # build the prompt once per batch
        prompt = (
            "You are a helpful assistant. Given the following full resume text, "
            f"generate {batch_size} useful question-answer pairs that a visitor might ask about the candidate. "
            "Return only a JSON array of objects with \"question\" and \"answer\" fields.\n\n"
            "Full resume text:\n\"\"\"\n"
            + resume_text +
            "\n\"\"\"\n"
        )

        response = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": "You generate concise QA pairs from resume text."},
                {"role": "user",   "content": prompt},
            ],
            response_format=QuestionAnswerPair,
            temperature=0.1
        )

        event = response.choices[0].message.parsed
        questions = event.question
        answers   = event.answer
        new_this_round = 0

        for i, question in enumerate(questions):
            if question not in seen_questions:
                seen_questions.add(question)
                results.append({
                    "question": question,
                    "answer":   answers[i]
                })
                new_this_round += 1

        num_fetched = len(questions)
        print(
            f"Batch fetched: {num_fetched} pairs → "
            f"{new_this_round} new → {len(results)}/{target_count}"
        )

        if new_this_round == 0:
            print("No new unique questions—stopping early to avoid infinite loop.")
            break

        time.sleep(pause_s)

    return results[:target_count]
