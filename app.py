from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
import PyPDF2
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
api_key="your key"
genai.configure(api_key=api_key)
text_splitter = CharacterTextSplitter(
    separator='\n',
    chunk_size=2000,
    chunk_overlap=200,
    length_function=len,
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = genai.GenerativeModel(
    model_name="gemini-1.5-flash"
)
resume_summary_template = """
Role: You are an AI Career Coach.

Task: Given the candidate's resume, provide a comprehensive summary that includes the following key aspects:

- Career Objective
- Skills and Expertise
- Professional Experience
- Educational Background
- Notable Achievements

Instructions:
Provide a concise summary of the resume, focusing on the candidate's skills, experience, and career trajectory. Ensure the summary is well-structured, clear, and highlights the candidate's strengths in alignment with industry standards.

Requirements:
{resume}

"""
def perform_qa(query):
    db = FAISS.load_local("vector_index", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    # rqa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    # result = rqa.invoke(query)
    # return result['result']
    # Get relevant documents
    docs = retriever.invoke(query)

    # Combine retrieved text
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = f"""
    You are an AI career coach.

    Below is context extracted from the candidate's resume and related documents:

    {context}

    Based on this context, please answer the following question:

    "{query}"

    If the answer is not present in the context, say "I could not find this information in the provided documents."
    """

    # Generate response with Gemini
    response = llm.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.3,
            max_output_tokens=600
        )
    )

    return response.text.strip()


app = Flask(__name__)

# File upload configuration
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text


resume_prompt = PromptTemplate(
    input_variables=["resume"],
    template=resume_summary_template,
)

# resume_analysis_chain = LLMChain(
#     llm=llm,
#     prompt=resume_prompt,
# )


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Extracted   the  text from the PDF
        resume_text = extract_text_from_pdf(file_path)
        splitted_text = text_splitter.split_text(resume_text)
        vectorstore = FAISS.from_texts(splitted_text, embeddings)
        vectorstore.save_local("vector_index")
        prompt = resume_summary_template.format(resume=resume_text)
        # print(proposal_text)
        # Run SWOT analysis using the LLM chain
        #resume_analysis = resume_analysis_chain.run(resume=resume_text)
        response = llm.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=800
            )
        )
        # Get the output text
        resume_analysis = response.text.strip()
        return render_template('results.html', resume_analysis=resume_analysis)


@app.route('/ask', methods=['GET', 'POST'])
def ask_query():
    if request.method == 'POST':
        query = request.form['query']
        result = perform_qa(query)
        return render_template('qa_results.html', query=query, result=result)
    return render_template('ask.html')


if __name__ == "__main__":
    app.run(debug=True)
