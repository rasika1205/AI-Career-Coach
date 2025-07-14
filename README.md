![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Build Passing](https://img.shields.io/badge/build-passing-brightgreen.svg)
![License](https://img.shields.io/badge/license-proprietary-lightgrey.svg)

# AI Career Coach

AI Career Coach is a web application that leverages advanced AI models to analyze resumes and provide career coaching. Users can upload their resumes in PDF format and receive a comprehensive summary highlighting their skills, experience, educational background, and achievements. The app also allows users to ask specific career-related questions based on their uploaded resume.

## Table of Contents

- [Features](#features)
- [Demo](#demo)
- [How It Works](#how-it-works)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [License](#license)

## Features

- **Resume Analysis:** Upload a PDF resume and get a structured summary:
  - Career Objective
  - Skills and Expertise
  - Professional Experience
  - Educational Background
  - Notable Achievements
  - **Interactive Q&A:** Ask career questions and get context-aware answers from the AI.

## Demo
<img width="1848" height="910" alt="Screenshot 2025-07-13 124954" src="https://github.com/user-attachments/assets/6ac3dd54-4deb-40a9-ac30-14894b7d6a56" />

## How It Works

1. **Upload Resume**: Users upload their PDF resume via a simple web interface.
2. **Text Extraction**: The system extracts text from the uploaded PDF.
3. **AI Processing**: The extracted text is processed using language models from Google Gemini and HuggingFace, with semantic search powered by FAISS.
4. **Summary Generation**: The AI generates a detailed and concise summary of the resume.
5. **Career Q&A**: Users can ask custom questions, and the AI provides informed answers based on the resume content.

## Technologies Used

- Python
- Flask (Web framework)
- PyPDF2 (PDF text extraction)
- LangChain, HuggingFace, FAISS (AI/ML and embeddings)
- Google Gemini (Generative AI for content creation)

## Getting Started

1. **Clone the repository:**
   ```
   git clone https://github.com/rasika1205/AI-Career-Coach.git
   cd AI-Career-Coach
   ```

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **API Key Setup:**
   - Create a `.env` file in the project root:
     ```bash
     touch .env
     ```
   - Add your Gemini API key to `.env`:
     ```env
     GEMINI_API_KEY=your_key_here
     ```

4. **Run the application:**
   ```
   python app.py
   ```
   - The app will be available at `http://127.0.0.1:5000/`.

## Usage

- Navigate to the web app in your browser.
- Upload your resume as a PDF.
- View the generated analysis and ask any follow-up questions.

## License

This project is **proprietary** and protected by copyright © 2025 Rasika Gautam.

You are welcome to view the code for educational or evaluation purposes (e.g., portfolio review by recruiters).  
However, you may **not copy, modify, redistribute, or claim this project as your own** under any circumstances — including in interviews or job applications — without written permission.

---

Feel free to explore the code.

_Developed with 💡 by [Rasika Gautam](https://github.com/rasika1205)_
