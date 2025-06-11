# Personal Profile Bot

A personal profile bot implementation with small language models for data generation and processing. This project provides tools to extract text from PDF resumes, generate question-answer pairs using OpenAI's language models, and create datasets for model training and distillation.

## Project Structure

```
personal-profile-bot/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── SmallLanguageModels.py
│   ├── data/
│   │   └── __init__.py
│   ├── utils/
│   │   └── __init__.py
│   └── config/
│       └── __init__.py
└── tests/
    └── __init__.py
```

## Features

- PDF resume text extraction using PyPDF2
- Advanced question-answer pair generation using OpenAI's GPT models with Teacher-LLM approach
- Balanced coverage rules for comprehensive QA dataset creation
- PyTorch Dataset and DataLoader for model training
- Pydantic models for structured data validation
- Modular project structure for easy maintenance
- Duplicate question filtering for unique Q&A datasets
- JSON data persistence with intelligent merging
- Knowledge distillation support for small language models

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from openai import OpenAI
from src.models.SmallLanguageModels import (
    extract_resume_text, 
    generate_unique_qa_pairs,
    save_qa_pairs
)

# Initialize OpenAI client
client = OpenAI()

# Extract text from PDF resume
resume_text = extract_resume_text("path/to/resume.pdf")

# Generate Q&A pairs using enhanced Teacher-LLM approach
qa_pairs = generate_unique_qa_pairs(
    client=client,
    resume_text=resume_text,
    target_count=300,
    batch_size=30
)

# Save results with duplicate handling
save_qa_pairs(qa_pairs, "qa_dataset.json")
```

### Dataset Creation for Training

```python
import json
from src.models.SmallLanguageModels import make_dataloader

# Load QA data
with open("qa_dataset.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

questions = [item["question"] for item in qa_data]
answers = [item["answer"] for item in qa_data]

# Create DataLoader for model training
loader = make_dataloader(
    questions, 
    answers, 
    tokenizer_name="facebook/opt-1.3b",
    batch_size=16,
    max_length=256
)

# Use with training loop
for batch in loader:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    # Your training logic here
```

### Environment Setup

Make sure to set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Development

This project follows a clean architecture with separation of concerns:

- **`src/models/`**: Contains language model implementations and data generation logic
- **`src/data/`**: Data processing and management utilities (future expansion)
- **`src/utils/`**: Utility functions and helper modules (future expansion)
- **`src/config/`**: Configuration management (future expansion)
- **`tests/`**: Test suites (future expansion)

### Code Quality

The project includes development dependencies for maintaining code quality:

- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking
- **Pytest**: Testing framework

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ tests/
```

## API Reference

### SmallLanguageModels Module

#### `extract_resume_text(pdf_path: str) -> str`

Extracts text content from a PDF resume file.

**Parameters:**
- `pdf_path`: Path to the PDF file

**Returns:**
- Extracted text as a string

#### `generate_unique_qa_pairs(client, resume_text, target_count=300, batch_size=30, model="gpt-4o", pause_s=1.0)`

Generates unique question-answer pairs from resume text using enhanced Teacher-LLM prompting with balanced coverage rules.

**Parameters:**
- `client`: OpenAI client instance
- `resume_text`: The resume text to generate Q&A pairs from
- `target_count`: Target number of unique Q&A pairs (default: 300)
- `batch_size`: Number of pairs to generate per API call (default: 30)
- `model`: OpenAI model to use (default: "gpt-4o")
- `pause_s`: Pause between API calls in seconds (default: 1.0)

**Returns:**
- List of dictionaries with "question" and "answer" keys

**Coverage Categories:**
- Career highlights & key skills (≥4 per batch)
- Project or research-specific details (≥4 per batch)
- Education, awards, or credentials (≥3 per batch)
- Work ethos / soft skills (≥2 per batch)
- Fun "about-you" ice-breakers (≥2 per batch)
- Random off-topic questions with fallback responses (≥4 per batch)

#### `save_qa_pairs(pairs, filename="qa_dataset.json")`

Save or update a JSON file containing QA pairs with intelligent duplicate handling.

**Parameters:**
- `pairs`: List of QA pairs to save
- `filename`: Output filename (default: "qa_dataset.json")

#### `QADistillationDataset`

PyTorch Dataset class for creating training datasets from QA pairs. Supports tokenization and formatting for knowledge distillation tasks.

**Parameters:**
- `questions`: List of questions
- `answers`: List of corresponding answers
- `tokenizer_name`: HuggingFace tokenizer name (default: "facebook/opt-1.3b")
- `max_length`: Maximum sequence length (default: 256)

#### `make_dataloader(questions, answers, ...)`

Creates a PyTorch DataLoader ready for training loops.

**Parameters:**
- `questions`: List of questions
- `answers`: List of corresponding answers
- `tokenizer_name`: HuggingFace tokenizer name (default: "facebook/opt-1.3b")
- `batch_size`: Batch size (default: 16)
- `max_length`: Maximum sequence length (default: 256)
- `shuffle`: Whether to shuffle the data (default: True)

**Returns:**
- PyTorch DataLoader for the QA dataset

#### `QuestionAnswerPair`

Pydantic model for validating question-answer pair responses from the API.

## Contributing

Please follow the existing code structure and maintain clean, readable code. Ensure all new features include appropriate tests and documentation.

## License

This project is for personal use and development purposes.
