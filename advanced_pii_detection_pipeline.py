import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.io import ReadFromText, WriteToText, ReadFromPubSub
from apache_beam.transforms.window import FixedWindows
from apache_beam.metrics import Metrics, MetricsFilter
from google.cloud import aiplatform, dlp_v2, pubsub_v1
import json
import argparse
from datetime import datetime
import time

class CombinedPIIDetector(beam.DoFn):
    def __init__(self, alert_topic):
        self.alert_topic = alert_topic
        self.dlp_request_count = Metrics.counter(self.__class__, 'dlp_request_count')
        self.vertex_request_count = Metrics.counter(self.__class__, 'vertex_request_count')
        self.dlp_latency = Metrics.distribution(self.__class__, 'dlp_latency_ms')
        self.vertex_latency = Metrics.distribution(self.__class__, 'vertex_latency_ms')

    def setup(self):
        self.dlp_client = dlp_v2.DlpServiceClient()
        aiplatform.init(project='your-project-id', location='us-central1')
        self.vertex_model = aiplatform.TextGenerationModel.from_pretrained("text-bison@001")
        self.publisher = pubsub_v1.PublisherClient()

    def process(self, elements, project):
        # Batch processing for Vertex AI
        vertex_inputs = [self.create_vertex_prompt(elem) for elem in elements]
        vertex_start_time = time.time()
        vertex_responses = self.vertex_model.batch_predict(instances=vertex_inputs)
        vertex_end_time = time.time()
        self.vertex_request_count.inc(len(elements))
        self.vertex_latency.update(int((vertex_end_time - vertex_start_time) * 1000))

        # Process each element
        for element, vertex_response in zip(elements, vertex_responses.predictions):
            if isinstance(element, bytes):
                element = element.decode('utf-8')
            
            try:
                data = json.loads(element)
                text_id, text_content = data['id'], data['text']
            except (json.JSONDecodeError, KeyError):
                text_id = datetime.now().isoformat()
                text_content = element

            # DLP API processing
            dlp_start_time = time.time()
            dlp_response = self.dlp_client.inspect_content(
                request={
                    "parent": f"projects/{project}",
                    "inspect_config": {
                        "info_types": [
                            {"name": "PERSON_NAME"},
                            {"name": "PHONE_NUMBER"},
                            {"name": "EMAIL_ADDRESS"},
                            {"name": "US_SOCIAL_SECURITY_NUMBER"},
                            {"name": "CREDIT_CARD_NUMBER"},
                            {"name": "STREET_ADDRESS"},
                        ],
                        "min_likelihood": dlp_v2.Likelihood.LIKELY,
                    },
                    "item": {"value": text_content}
                }
            )
            dlp_end_time = time.time()
            self.dlp_request_count.inc()
            self.dlp_latency.update(int((dlp_end_time - dlp_start_time) * 1000))

            # Combine DLP and Vertex AI results
            dlp_findings = [
                {
                    "info_type": finding.info_type.name,
                    "likelihood": dlp_v2.Finding.Likelihood(finding.likelihood).name,
                    "quote": finding.quote
                }
                for finding in dlp_response.result.findings
            ]

            try:
                vertex_analysis = json.loads(vertex_response)
            except json.JSONDecodeError:
                vertex_analysis = {"error": "Failed to parse Vertex AI response"}

            combined_result = {
                'id': text_id,
                'text': text_content,
                'dlp_findings': dlp_findings,
                'vertex_analysis': vertex_analysis,
                'has_pii': bool(dlp_findings) or vertex_analysis.get('has_pii', False)
            }

            # Send real-time alert if high-risk PII is detected
            if self.is_high_risk_pii(combined_result):
                self.send_alert(combined_result)

            yield combined_result

    def create_vertex_prompt(self, element):
        if isinstance(element, bytes):
            element = element.decode('utf-8')
        
        try:
            data = json.loads(element)
            text_content = data['text']
        except (json.JSONDecodeError, KeyError):
            text_content = element

        return f"""
        Analyze the following text for any potential Personally Identifiable Information (PII).
        Identify and list any PII elements found, such as names, addresses, phone numbers, 
        email addresses, social security numbers, credit card numbers, etc.
        If no PII is found, state that the text is clear of PII.
        Format the response as JSON with 'has_pii' (boolean) and 'pii_elements' (list) fields.

        Text: {text_content}

        JSON Response:
        """

    def is_high_risk_pii(self, result):
        high_risk_types = ['US_SOCIAL_SECURITY_NUMBER', 'CREDIT_CARD_NUMBER']
        return any(finding['info_type'] in high_risk_types for finding in result['dlp_findings'])

    def send_alert(self, result):
        alert_message = json.dumps({
            'message': 'High-risk PII detected',
            'id': result['id'],
            'detected_pii': [finding['info_type'] for finding in result['dlp_findings']]
        })
        self.publisher.publish(self.alert_topic, alert_message.encode('utf-8'))

def run(args):
    pipeline_options = PipelineOptions([
        f'--runner={args.runner}',
        f'--project={args.project}',
        f'--region={args.region}',
        f'--temp_location=gs://{args.bucket}/temp',
        f'--job_name=advanced-pii-detection-{datetime.now().strftime("%Y%m%d-%H%M%S")}',
        '--streaming' if args.streaming else '--no-streaming'
    ])

    with beam.Pipeline(options=pipeline_options) as p:
        if args.streaming:
            input_data = (p | 'ReadFromPubSub' >> ReadFromPubSub(subscription=args.subscription)
                            | 'WindowInto' >> beam.WindowInto(FixedWindows(60)))  # 1-minute fixed windows
        else:
            input_data = p | 'ReadFromGCS' >> ReadFromText(args.input)

        (input_data 
         | 'BatchElements' >> beam.BatchElements(min_batch_size=10, max_batch_size=100)
         | 'DetectPII' >> beam.ParDo(CombinedPIIDetector(args.alert_topic), args.project)
         | 'FormatOutput' >> beam.Map(json.dumps)
         | 'WriteResults' >> WriteToText(args.output)
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True, help='GCP Project ID')
    parser.add_argument('--region', default='us-central1', help='GCP region')
    parser.add_argument('--bucket', required=True, help='GCS bucket for temp files and output')
    parser.add_argument('--runner', default='DataflowRunner', help='Beam runner (DataflowRunner or DirectRunner)')
    parser.add_argument('--streaming', action='store_true', help='Enable streaming pipeline')
    parser.add_argument('--subscription', help='Pub/Sub subscription (for streaming)')
    parser.add_argument('--input', help='Input GCS path (for batch)')
    parser.add_argument('--output', required=True, help='Output GCS path')
    parser.add_argument('--alert_topic', required=True, help='Pub/Sub topic for real-time alerts')
    args = parser.parse_args()

    run(args)
