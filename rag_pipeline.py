from sentence_transformers import SentenceTransformer
from google import genai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from eval import run_evaluations

file_path = "Q3_Insurance_Report.md"


with open(file_path, "r", encoding="utf-8") as file:
    markdown_text = file.read()



raw_chunks = markdown_text.strip().split("\n\n")

processed_chunks = [chunk.strip() for chunk in raw_chunks if chunk.strip()]

model = SentenceTransformer('all-MiniLM-L6-v2')

vectors = model.encode(processed_chunks)

test_prompts = [
    "What was the total Gross Written Premium (GWP) for Q3 2025, and what was the year-over-year growth percentage?",
    "How much money did Project Clear-Path save the company in loss adjustment expenses this quarter?",
    "What is the current asset allocation mix of the investment portfolio?",
    "Which specific weather events caused the spike in catastrophic losses this quarter?",
    "Why is the Workers' Compensation line of business performing so well right now?",
    "What specific issues forced the actuarial team to adjust the reserves for the 2022 and 2023 accident years?",
    "What is the company's target goal for the overall combined ratio by the end of the year?",
    "I need an update on our regulatory hurdles. What is preventing us from growing our auto business on the West Coast?",
    "Are we utilizing any artificial intelligence or machine learning to speed up how fast we pay out claims?",
    "How did the Treasury team handle the bond portfolio to protect us from interest rate changes?"
]

# get the user's prompt 
# encode it 
prompt_vectors = model.encode(test_prompts)


final_prompts = []

client = genai.Client()


def findRevalantData(vector) -> str:
    # compare the user's prompt embeeded to all the rest of the processed chunks
    # if the simlailrity score is higher than the current one then update it 
    best_score = -1.0
    best_index = -1

    vector = vector.reshape(1, -1)
    for i,chunk in enumerate(vectors):
        chunk_2d = chunk.reshape(1, -1)

        score = cosine_similarity(vector, chunk_2d)[0][0]

        if score > best_score:
            best_score = score
            best_index = i

    return processed_chunks[best_index]


def sendPrompt(system_prompt: str) -> str:
    gemini_response = client.models.generate_content(
        model = "gemini-3.1-flash-lite-preview",
        contents=system_prompt
    )

    return gemini_response.text

context_data = []

for i, prompt_vector in enumerate(prompt_vectors):
    # add the data of the index with the closest data 
    data = findRevalantData(prompt_vector)
    context_data.append(data)
    user_question = test_prompts[i]

    # send the updated system prompt
    system_prompt = f"""
    You are a highly accurate corporate assistant. 
    Answer the user's question using ONLY the provided context below. 
    If the answer is not contained in the context, say "I don't have enough information."

    Context:
    {data}

    Question:
    {user_question}
    """

    final_prompts.append(system_prompt)

all_responses = []

# get the results 
for i,system_prompt in enumerate(final_prompts):
    response = sendPrompt(system_prompt=system_prompt)

    print(f"Finished {i} response: {response}")
    all_responses.append({
        "Question": test_prompts[i],
        "Data": context_data[i],
        "Response": response
    })

# evaluate the results
run_evaluations(all_responses)