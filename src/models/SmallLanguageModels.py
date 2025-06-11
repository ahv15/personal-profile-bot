#!/usr/bin/env python
# coding: utf-8

"""
Small Language Models module for personal profile bot.

This module provides functionality for extracting text from PDF resumes,
generating question-answer pairs using OpenAI's language models, and creating
datasets for model training and distillation.
"""

import json
import os
import time
from typing import List, Dict, Any

import torch
from PyPDF2 import PdfReader
from pydantic import BaseModel
from openai import OpenAI
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


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
    target_count: int = 300,
    batch_size: int = 30,
    model: str = "gpt-4o",
    pause_s: float = 1.0
) -> List[Dict[str, str]]:
    """
    Generate unique question-answer pairs from resume text using enhanced Teacher-LLM prompting.
    
    This function uses a sophisticated prompt that ensures balanced coverage across different
    categories and includes off-topic questions with appropriate fallback responses.
    
    Args:
        client (OpenAI): OpenAI client instance
        resume_text (str): The resume text to generate QA pairs from
        target_count (int, optional): Target number of unique QA pairs. Defaults to 300.
        batch_size (int, optional): Number of pairs to generate per API call. Defaults to 30.
        model (str, optional): OpenAI model to use. Defaults to "gpt-4o".
        pause_s (float, optional): Pause between API calls in seconds. Defaults to 1.0.
        
    Returns:
        List[Dict[str, str]]: List of dictionaries with "question" and "answer" keys
    """
    seen_questions = set()
    results = []

    while len(results) < target_count:
        # Build enhanced prompt with detailed instructions for balanced coverage
        prompt = f"""
            You are the **Teacher-LLM** responsible for distilling your knowledge into a
            much smaller student model.

            ──────────────────────────── 1  Task Context ────────────────────────────
            Below is the full résumé and portfolio of *Harshit Alluri*, delimited by
            <RESUME></RESUME> tags.

            ──────────────────────────── 2  Output Format ───────────────────────────
            Return **EXACTLY {batch_size} objects** in a **top-level JSON array**.
            Each object has keys:
              • "question": a plausible visitor / recruiter / random-general query  
              • "answer":   a concise reply, ≤ 45 words

            ──────────────────────────── 3  Coverage Rules ──────────────────────────
            Generate a balanced mix every batch:
              a. Career highlights & key skills                ≥ 4  
              b. Project- or research-specific details         ≥ 4  
              c. Education, awards, or credentials             ≥ 3  
              d. Work ethos / soft skills                      ≥ 2  
              e. Fun "about-you" ice-breakers                  ≥ 2  
              f. **Random off-topic** (weather, general knowledge, simple math,
                 small-talk) ≥ 4.  
                 • For *all* off-topic items: **ALWAYS** answer with the fixed fallback  
                  string exactly as shown:  
                  `"I'm sorry, I don't have information about that. Please feel free to ask about my work or background!"`

            Extra diversity directives  
              • Vary interrogatives: *What / Which / How did …?*, *Could you share …?*,  
                *Tell me about …*, *In which year …?*, etc.  
              • Mix formal & conversational tones.  
              • Avoid near-duplicates across the batch **and** across previous batches  
                (assume you store seen questions).  
              • No résumé line numbers, no external URLs.

            ──────────────────────────── 4  Few-Shot Examples (10) ───────────────────
            ```json
            [
              {{ "question": "Which runtime framework did Harshit leverage to cut cold-start latency in his LLM autoscaler?",
                 "answer": "He used Ray Serve, dynamically spinning up GPU pods and cutting cold-start time by 60 %." }},

              {{ "question": "In the adaptive model-multiplexing layer, how were accuracy and latency balanced?",
                 "answer": "A lightweight policy net routed queries either to a fast 7-B model or a larger 34-B model, trimming 15 ms latency while boosting accuracy 8 %." }},

              {{ "question": "Where did Harshit complete his undergraduate studies, and with what GPA?",
                 "answer": "Amrita Vishwa Vidyapeetham (B.Tech. in AI) with a 9.77/10 GPA." }},

              {{ "question": "What leadership trait has Harshit demonstrated when integrating multi-agent pipelines?",
                 "answer": "He's known for proactive cross-team communication—writing clear RFCs, documenting edge cases, and mentoring junior contributors." }},

              {{ "question": "If Harshit weren't working in ML, which musical instrument would he love to master?",
                 "answer": "I'm sorry, but that information isn't in my résumé." }},

              {{ "question": "Is Atlanta expecting thunderstorms this weekend?",
                 "answer": "I'm sorry, I don't have information about that. Please feel free to ask about my work or background!" }},

              {{ "question": "Who wrote the novel *One Hundred Years of Solitude*?",
                 "answer": "I'm sorry, I don't have information about that. Please feel free to ask about my work or background!" }},

              {{ "question": "What is the square root of 361?",
                 "answer": "I'm sorry, I don't have information about that. Please feel free to ask about my work or background!" }},

              {{ "question": "Do you prefer thin-crust or deep-dish pizza?",
                 "answer": "I'm sorry, I don't have information about that. Please feel free to ask about my work or background!" }},

              {{ "question": "Could you outline Harshit's contribution to the MedSim differential diagnosis engine?",
                 "answer": "He built a BioBERT + Graph Attention Network to extract temporal relations, powering symptom-progression graphs used in 40+ medical colleges." }}
            ]
            ──────────────────────────── 5 Résumé ────────────────────────────
            <RESUME>
            {resume_text}
            </RESUME>
        """

        # Make API call with structured output
        response = client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": "You generate concise QA pairs from resume text."},
                {"role": "user", "content": prompt},
            ],
            text_format=QuestionAnswerPair,
            temperature=1
        )

        # Parse response and extract QA pairs
        event = response.output_parsed
        questions = event.question
        answers = event.answer
        new_this_round = 0
        
        print(questions)
        print(answers)
        
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


def save_qa_pairs(pairs: List[Dict[str, str]], filename: str = "qa_dataset.json") -> None:
    """
    Save or update a JSON file containing QA pairs.
    
    If the file doesn't exist, it creates it. If it exists, it loads existing pairs,
    appends new ones (avoiding duplicates), and rewrites the file.
    
    Args:
        pairs (List[Dict[str, str]]): List of QA pairs to save
        filename (str, optional): Output filename. Defaults to "qa_dataset.json".
    """
    # Load existing data if file exists
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = []
    
    # Build a set of existing questions to avoid duplicates
    existing_questions = {item["question"] for item in existing}
    
    # Filter new pairs that are not in existing
    new_items = [item for item in pairs if item["question"] not in existing_questions]
    
    # Update list
    updated = existing + new_items
    
    # Write back to file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(new_items)} new items (total {len(updated)}) to '{filename}'")


class QADistillationDataset(Dataset):
    """
    A Dataset that takes parallel lists of questions and answers,
    builds "Q: ... A:" prompts, and tokenizes inputs+targets in one call.
    
    This dataset is designed for training small language models using
    knowledge distillation from larger models.
    """
    
    def __init__(self,
                 questions: List[str],
                 answers: List[str],
                 tokenizer_name: str = "facebook/opt-1.3b",
                 max_length: int = 256):
        """
        Initialize the QA dataset.
        
        Args:
            questions (List[str]): List of questions
            answers (List[str]): List of corresponding answers
            tokenizer_name (str, optional): HuggingFace tokenizer name. Defaults to "facebook/opt-1.3b".
            max_length (int, optional): Maximum sequence length. Defaults to 256.
        """
        assert len(questions) == len(answers), "Questions and answers lists must be the same length"
        
        self.questions = questions
        self.answers = answers
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        
        # Add pad token if not present
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a tokenized QA pair.
        
        Args:
            idx (int): Index of the QA pair
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing input_ids, attention_mask, and labels
        """
        q = self.questions[idx].strip()
        a = self.answers[idx].strip()
        prompt = f"Q: {q}\nA:"
        
        tokenized = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            text_target=a
        )

        input_ids = tokenized.input_ids.squeeze(0)
        attention_mask = tokenized.attention_mask.squeeze(0)
        labels = tokenized.labels.squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def make_dataloader(questions: List[str], 
                   answers: List[str],
                   tokenizer_name: str = "facebook/opt-1.3b",
                   batch_size: int = 16,
                   max_length: int = 256,
                   shuffle: bool = True) -> DataLoader:
    """
    Create a DataLoader ready for Trainer or manual training loops.
    
    Args:
        questions (List[str]): List of questions
        answers (List[str]): List of corresponding answers
        tokenizer_name (str, optional): HuggingFace tokenizer name. Defaults to "facebook/opt-1.3b".
        batch_size (int, optional): Batch size. Defaults to 16.
        max_length (int, optional): Maximum sequence length. Defaults to 256.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        
    Returns:
        DataLoader: PyTorch DataLoader for the QA dataset
    """
    dataset = QADistillationDataset(
        questions, answers,
        tokenizer_name=tokenizer_name,
        max_length=max_length
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True
    )
