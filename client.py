import requests
import json


def consume_stream():
    url = "http://127.0.0.1:8000/generate"

    payload = {
        "doc_id": "Q3_Insurance_Report.md",
        "user_prompt": "What was the total Gross Written Premium (GWP) for Q3 2025, and what was the year-over-year growth percentage?",
        "stream": True
    }

    print("SEnding request to backend... \n")
    print("AI Response: ", end="")

    with requests.post(url, json=payload, stream=True) as response:
        for line in response.iter_lines(decode_unicode=True):
            # Ignore empty lines (keep-alives) and look for our data payload
            if line and line.startswith("data: "):
                
                # Strip away the "data: " prefix to get the raw JSON string
                json_str = line[6:]

                data = json.loads(json_str)

                print(data["token"], end="", flush=True)

                if data.get("is_final"):
                    break

    print("\n Stream complete")


if __name__ == "__main__":
    consume_stream()
