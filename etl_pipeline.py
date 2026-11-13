import os
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv # optional if using .env file

load_dotenv()

# 1. Initialize OpenAI client
# Ensure your OPENAI_API_KEY is set in your environment variables
client = OpenAI(api_key=os.environ.get("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"))

def clean_row_with_openai(dirty_data):
    """Clean one student record using OpenAI."""

    prompt = f"""
    Clean this student enrollment record.

    Rules:
    - Keep exactly these fields:
      Student_ID, Name, Email, Phone, Course, Fee_Paid, City, Enrolled_Date
    - Trim unnecessary spaces.
    - Standardize names and email formatting.
    - Do not invent missing information.
    - Preserve the original meaning of every field.
    - Return only a valid JSON object.

    Rules!
- Name: Title Case (Manoj K)
- Email: if invalid set INVALID_EMAIL
- Phone: if less than 10 digits set MISSING
- Course: only Python or Machine Learning or Data
SeaneR
- Fee_Paid: yes/YES = true, no/NO = false
- City: Title Case
empty • UNKNOWN
- Enrolled_Date: convert to YYYY-MM-D0
Return ONLY raw JSON. No explanation. No markdown.

    Student record:
    {json.dumps(dirty_data)}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You clean student enrollment data and return valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"}
        )

        cleaned_data = json.loads(
            response.choices[0].message.content
        )
        return cleaned_data

    except Exception as error:
        print(f"Error cleaning record: {error}")
        return None


if __name__ == "__main__":
    INPUT_CSV = "student_enrollment.csv"
    OUTPUT_CSV = "clean_students.csv"

    
def run_etl(input_file, output_file):
    print(f"Loading {input_file}...")

    df = pd.read_csv(input_file, dtype=str).fillna("")
    print(f"Loaded {len(df)} student records.")

    cleaned_rows = []

    for index, row in df.iterrows():
        dirty_string = row.to_dict()

        print(f"Cleaning student {index + 1}/{len(df)}...")

        cleaned_data = clean_row_with_openai(dirty_string)

        if cleaned_data is not None:
            cleaned_rows.append(cleaned_data)
        else:
            cleaned_rows.append(dirty_string)

    cleaned_df = pd.DataFrame(cleaned_rows)
    cleaned_df.to_csv(output_file, index=False)

    print("ETL completed successfully!")
    print(f"Input rows: {len(df)}")
    print(f"Output rows: {len(cleaned_df)}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    INPUT_CSV = "student_enrollment.csv"
    OUTPUT_CSV = "clean_students.csv"

    run_etl(INPUT_CSV, OUTPUT_CSV)
