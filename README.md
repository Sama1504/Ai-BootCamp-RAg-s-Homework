# AI Bootcamp in India - Homework 3

## Overview

This project implements a RAG (Retrieval-Augmented Generation) application using the Google Gemini API. It was originally designed for OpenAI's API but has been adapted due to token limitations and system resource constraints.

## Background

Initially, this project was set up to use OpenAI's API. However, we encountered several challenges:

1. OpenAI API token limitations (5000 tokens) were insufficient for this project.
2. Creating the output.json file was time-consuming and caused system overheating.
3. System errors occurred twice during processing.

To address these issues, we switched to the Google Gemini API, which offers a free tier with higher token limits.

## Prerequisites

- Python 3.10+
- Google Gemini API Key

## Setup

1. Clone this repository:
   ```sh
   git clone <repository-url>
   cd <repository-name>
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root and add your Google Gemini API key:
   ```sh
   GOOGLE_API_KEY=your_google_gemini_api_key_here
   ```

## Known Issues and Troubleshooting

1. Deprecated Imports:
   You may see warnings about deprecated imports from `langchain`. To resolve this, update the imports in `rag_implementation.py`:

   ```python
   from langchain_community.vectorstores import Chroma
   from langchain_community.embeddings import GooglePalmEmbeddings
   from langchain_community.llms import GooglePalm
   ```

2. API Key Not Found:
   If you encounter an error like "Did not find google_api_key", ensure that:
   - Your `.env` file is in the correct location (project root).
   - The API key in the `.env` file is correct and properly formatted.
   - You're running the script from the project root directory.

3. Transcript Issues:
   The transcript of the one-hour video may sometimes cause issues. If you encounter problems related to the transcript, try:
   - Checking the `docs/intro-to-llms-karpathy.txt` file for any formatting issues.
   - Splitting the transcript into smaller chunks if it's too large.

## Running the Project

1. Implement the RAG system:
   ```sh
   python rag_implementation.py
   ```

2. Generate answers to the question list:
   ```sh
   python generate_answers.py
   ```

3. Evaluate the results:
   ```sh
   python eval.py path/to/your/output.json
   ```

## Troubleshooting

If you encounter the error "TypeError: str expected, not NoneType" when setting the API key, try modifying the `rag_implementation.py` file:
