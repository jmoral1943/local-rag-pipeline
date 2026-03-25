from google import genai
import json
import time
import csv
from pydantic import BaseModel, Field


class EvaluationScore(BaseModel):
    tone: int = Field(description="Score 1-5. 1 = Hostile/Rude. 3 = Robotic/Blunt. 5 = Professional and Helpful.")
    relevance: int = Field(description="Score 1-5. 1 = Complete hallucination. 3 = Partially correct but missing details. 5 = Perfectly accurate based ONLY on the document.")


client = genai.Client()

def run_evaluations(rag_results, csv_filename="eval_results.csv"):
    print(f"Starting evaluation on {len(rag_results)} records...")
    csv_data = []

    for item in rag_results:
        question = item.get("Question")
        data_context = item.get("Data")
        response = item.get("Response")

        content = f"Document: {json.dumps(data_context)}\n\nQuestion: {question}\nThe response from an assistant is: {response}\nGrade it on a scale of 1 to 5."    

        gemini_response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview", contents= content,
        config= {
            "response_mime_type": "application/json",
            "response_json_schema": EvaluationScore.model_json_schema(),
            }
        )
        
        scores = EvaluationScore.model_validate_json(gemini_response.text)

        print(f"Evaluated: Tone={scores.tone}, Relevance={scores.relevance}")

        # Append everything, including the new 'Data' field, to our CSV array
        csv_data.append({
            "Question": question,
            "Data": data_context, 
            "Response": response,
            "Tone_score": scores.tone,
            "Relevance_score": scores.relevance
        })
        
        # Respect API rate limits
        time.sleep(5)

    # Write the array to a CSV file
    # We added "Data" to the fieldnames to match your request
    fieldnames = ["Question", "Data", "Response", "Tone_score", "Relevance_score"]
    
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)

    print(f"\nSuccess! Evaluated {len(csv_data)} interactions and saved to {csv_filename}")