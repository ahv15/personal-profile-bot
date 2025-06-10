# Personal Profile Bot

A personal profile bot implementation with small language models for data generation and processing. This project provides tools to extract text from PDF resumes and generate question-answer pairs using OpenAI's language models.

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
- Question-answer pair generation using OpenAI's GPT models
- Pydantic models for structured data validation
- Modular project structure for easy maintenance
- Duplicate question filtering for unique Q&A datasets

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from openai import OpenAI
from src.models.SmallLanguageModels import extract_resume_text, generate_unique_qa_pairs

# Initialize OpenAI client
client = OpenAI()

# Extract text from PDF resume
resume_text = extract_resume_text("path/to/resume.pdf")

# Generate Q&A pairs
qa_pairs = generate_unique_qa_pairs(
    client=client,
    resume_text=resume_text,
    target_count=100,
    batch_size=10
)

# Save results
import json
with open("qa_dataset.json", "w") as f:
    json.dump(qa_pairs, f, indent=2)
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

#### `generate_unique_qa_pairs(client, resume_text, target_count=2000, batch_size=100, model="gpt-4o", pause_s=1.0)`

Generates unique question-answer pairs from resume text using OpenAI's API.

**Parameters:**
- `client`: OpenAI client instance
- `resume_text`: The resume text to generate Q&A pairs from
- `target_count`: Target number of unique Q&A pairs (default: 2000)
- `batch_size`: Number of pairs to generate per API call (default: 100)
- `model`: OpenAI model to use (default: "gpt-4o")
- `pause_s`: Pause between API calls in seconds (default: 1.0)

**Returns:**
- List of dictionaries with "question" and "answer" keys

#### `QuestionAnswerPair`

Pydantic model for validating question-answer pair responses from the API.

## Contributing

Please follow the existing code structure and maintain clean, readable code. Ensure all new features include appropriate tests and documentation.

## License

This project is for personal use and development purposes.
