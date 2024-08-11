from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict
import os

def summarize_text(project_id, location, model_name, text):
    # Initialize Vertex AI client
    aiplatform.init(project=project_id, location=location)

    # Load the pre-trained text summarization model
    model = aiplatform.Model(model_name=model_name)

    # Prepare the input data
    instance = predict.instance.TextPredictionInstance(
        content=text,
    ).to_value()

    # Make the prediction
    prediction = model.predict([instance])

    # Extract and return the summary
    summary = prediction.predictions[0]
    return summary

def process_servicenow_items(items):
    project_id = "your-project-id"
    location = "us-central1"
    model_name = "text-bison@001"

    summaries = []
    for item in items:
        # Combine relevant fields for summarization
        full_text = f"Title: {item['short_description']}\n\nDescription: {item['description']}"
        
        # Generate summary
        summary = summarize_text(project_id, location, model_name, full_text)
        
        summaries.append({
            "item_id": item["sys_id"],
            "original_text": full_text,
            "summary": summary
        })
    
    return summaries

# Example usage
servicenow_items = [
    {
        "sys_id": "incident001",
        "short_description": "Email service is down",
        "description": "Users are reporting that they cannot send or receive emails. The issue started at 9:00 AM and is affecting all departments. IT team is investigating the root cause."
    },
    {
        "sys_id": "change001",
        "short_description": "Upgrade database server",
        "description": "Scheduled maintenance to upgrade the main database server from version 10.5 to 11.2. This will improve performance and add new features. Downtime is expected to be 2 hours."
    }
]

results = process_servicenow_items(servicenow_items)

for result in results:
    print(f"Item ID: {result['item_id']}")
    print(f"Original Text:\n{result['original_text']}\n")
    print(f"Summary:\n{result['summary']}\n")
    print("-" * 50)
