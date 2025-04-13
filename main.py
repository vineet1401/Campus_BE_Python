from flask import Flask, request, jsonify
import os
import logging
from pypdf import PdfReader
import google.generativeai as genai
from flask_cors import CORS
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=api_key)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)


@app.route("/process", methods=["POST"])
def evaluate_resume_against_jd():
    """Handles resume PDF and JD, and returns ATS score & missing skills"""
    try:
        # Validate inputs
        if 'pdf_doc' not in request.files:
            return jsonify({"error": "Resume PDF file not provided"}), 400
        if 'job_description' not in request.form:
            logging.error(f"Error during processing: ")
            return jsonify({"error": "Job description not provided"}), 400

        doc = request.files['pdf_doc']
        job_description = request.form.get('job_description')

        if not doc.filename.endswith('.pdf'):
            return jsonify({"error": "Only PDF files are allowed"}), 400

        # Step 1: Extract text from resume
        resume_text = extract_text_from_pdf(doc)
        if not resume_text:
            return jsonify({"error": "Failed to extract text from resume"}), 500

        # Step 2: Extract structured resume JSON
        resume_json_raw = extract_structured_resume(resume_text)
        if not resume_json_raw:
            return jsonify({"error": "Failed to parse resume into structured JSON"}), 500

        resume_json = json.loads(resume_json_raw)

        # Step 3: Calculate ATS score & missing skills
        ats_result_raw = calculate_ats_score(json.dumps(resume_json), job_description)
        if not ats_result_raw:
            return jsonify({"error": "Failed to calculate ATS score"}), 500

        ats_result = json.loads(ats_result_raw)
        ats_result["resume_data"] = resume_json  # Optional: include parsed resume data

        return jsonify(ats_result)

    except Exception as e:
        logging.error(f"Unexpected error in /ats-evaluate: {e}")
        return jsonify({"error": "Internal server error"}), 500


# -------------- Utility Functions ----------------

def extract_text_from_pdf(file):
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages[:5]:  # Limit to first 5 pages
            text += page.extract_text() or ""
        return text
    except Exception as e:
        logging.error(f"Error reading PDF: {e}")
        return ""


def extract_structured_resume(text):
    prompt = f"""
        Extract structured resume data from the following text. Return valid JSON only:

        {{
            "full_name": "",
            "email": "",
            "github": "",
            "linkedin": "",
            "employment": "",
            "technical_skills": [],
            "phone": "",
            "address": "",
            "profile": ""
        }}

        Resume Text:
        {text}
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt, stream=True)
        return "".join(chunk.text for chunk in response).replace("```json", "").replace("```", "").strip()
    except Exception as e:
        logging.error(f"Error in extract_structured_resume: {e}")
        return None


def calculate_ats_score(resume_json, job_description):
    prompt = f"""
        Compare the resume JSON below with the job description. Return a JSON like (ats_score should be out of 100):
        {{
            "ats_score": 0,
            "missing_skills": []
        }}

        Resume JSON:
        {resume_json}

        Job Description:
        {job_description}
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt, stream=True)
        return "".join(chunk.text for chunk in response).replace("```json", "").replace("```", "").strip()
    except Exception as e:
        logging.error(f"Error in calculate_ats_score: {e}")
        return None


# ---------------------- Main ----------------------

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
