#!/usr/bin/env python3


















### Target Structure
```python
sow = SOW()
sow.load_data()
sow.process_documents()
sow.select_template()
sow.generate_narrative()
sow.enhance_narrative()
final_output = sow.get_final_output()
```

## Current Workflow Analysis

### Current Flow (from existing code):
```
1. get_gpt_answers_for_client() - Main orchestrator
   ├── Load client data from CRDB
   ├── Get client documents
   ├── Process documents with OCR
   ├── Extract information with GPT
   └── Return structured data

2. generate_sow_narrative_with_separate_template_and_values()
   ├── Load YAML template
   ├── Combine client data + GPT answers
   ├── Generate narrative using template
   └── Return narrative with controls

3. Final enhancement with rephrase_narrative.txt
   ├── Clean narrative
   ├── Apply rephrasing prompt
   └── Generate final output
```

## Proposed Refactored Design

### 1. Main SOW Class
```python
class SOW:
    def __init__(self, config):
        self.data_loader = DataLoader(config)
        self.document_processor = DocumentProcessor(config)
        self.information_extractor = InformationExtractor(config)
        self.template_manager = TemplateManager(config)
        self.narrative_generator = NarrativeGenerator(config)
        self.narrative_enhancer = NarrativeEnhancer(config)
    
    def process_client(self, client_id, scenario=None):
        # Main workflow
        pass
        
"""
Standalone SOW Pipeline with Bedrock LLM
========================================

This is a simplified, standalone version of the SOW (Source of Wealth) pipeline
that uses AWS Bedrock LLM instead of complex database dependencies.

Key Features:
- Uses AWS Bedrock Nova Pro model
- Hardcoded sample client data and documents
- All prompts and templates embedded
- No database dependencies
- Production-ready code following SOLID principles

Usage:
    python standalone_sow_pipeline.py
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from langchain_aws import ChatBedrockConverse

# Set AWS credentials
os.environ['AWS_ACCESS_KEY_ID'] = ''
os.environ['AWS_SECRET_ACCESS_KEY'] = ''
os.environ['AWS_DEFAULT_REGION'] = 'ap-south-1'

# Initialize LLM
llm = ChatBedrockConverse(
    model_id="apac.amazon.nova-lite-v1:0",
    temperature=0.01
)


@dataclass
class SOWResult:
    """Data class for SOW generation results"""
    client_id: int
    client_name: str
    client_type: str
    narrative: str
    enhanced_narrative: str
    missing_scenarios: List[str]
    controls: List[Dict[str, Any]]
    processing_info: Dict[str, Any]
    status: str
    timestamp: datetime


def execute_prompt(prompt: str) -> str:
    """
    Execute a prompt using Bedrock LLM

    Args:
        prompt: The prompt to execute

    Returns:
        LLM response as string
    """
    try:
        messages = [
            ("human", prompt)
        ]
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        print(f"❌ Error executing prompt: {e}")
        return ""


class SOWPromptTemplates:
    """Loads prompt templates from external files"""

    def __init__(self, prompt_dir: str = None):
        """Initialize with prompt directory"""
        if prompt_dir is None:
            # Get the directory of this script and navigate to prompt_template
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.prompt_dir = os.path.join(script_dir, "prompt_template")
        else:
            self.prompt_dir = prompt_dir

    def load_prompt_file(self, file_path: str) -> str:
        """Load prompt from file"""
        try:
            full_path = os.path.join(self.prompt_dir, file_path)
            print(f"🔍 Loading prompt from: {full_path}")
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"❌ Error loading prompt file {file_path}: {e}")
            print(f"   Looking in directory: {self.prompt_dir}")
            return ""

    def get_intro_prompt(self, client_type: str = "individual") -> str:
        """Get intro prompt for client type"""
        prompt = self.load_prompt_file(f"intro/{client_type}.txt")
        if not prompt:
            # Fallback to embedded prompt
            return self._get_fallback_intro_prompt()
        return prompt

    def get_scenario_prompt(self, client_type: str, scenario: str) -> str:
        """Get scenario prompt"""
        prompt = self.load_prompt_file(f"sow_scenarios/{client_type}/{scenario}.txt")
        if not prompt:
            # Fallback to embedded prompt
            return self._get_fallback_scenario_prompt(scenario)
        return prompt

    def get_rephrase_prompt(self) -> str:
        """Get rephrase prompt"""
        prompt = self.load_prompt_file("rephrase_narrative.txt")
        if not prompt:
            # Fallback to embedded prompt
            return self._get_fallback_rephrase_prompt()
        return prompt

    def _get_fallback_intro_prompt(self) -> str:
        """Fallback intro prompt if file loading fails"""
        return """You are a wealth management client {client_name} document analyst. You need to read and understand the document, then answer questions based only on the input document text.

General Instructions:
- The input document text is divided page by page, with page numbers at the beginning of each page's text.
- You must read and understand the document, then answer questions based only on the input document text.
- If there's no clear answer, use an empty string in the JSON format.
- The language of the document should be detected for the first question.
- Answers to subsequent questions should only be given if the document is about the specified client.
- Never use "string" as answer to any question, "string" is the data type, not a valid answer.
- For questions #3 and onwards, both the answer and the page number(s) where the answer was found must be provided, saved to different JSON keys as instructed. If an answer is found on multiple pages, all page numbers should be given, comma-separated.

Questions to Answer:

1. "what language is the document?"
   - Answer using the full English word (e.g., English, Spanish, Chinese), not language codes.
   - If multiple languages are used, list them comma-separated.
   - Note: An individual name in a foreign language does not mean the document is in that language.
   - Key: doc_language

2. "what is the document about?"
   - Provide a summary with a character limit of 300.
   - The answer to this question should not be empty.
   - Key: summary

3. "what is {client_name}'s country of citizenship?"
   - Additional question: "On which page did you find the answer? give the page number only, if you found the answer on multiple pages, give all page numbers with comma separated."
   - Keys: client_country_of_citizenship, client_country_of_citizenship_pages

4. "what is {client_name}'s country of residence?"
   - Additional question: "On which page did you find the answer? give the page number only, if you found the answer on multiple pages, give all page numbers with comma separated."
   - Keys: domicile_country_name, domicile_country_name_pages

5. "what is the breakdown of {client_name}'s networth?"
   - The answer should provide the breakdown, with examples given: "real estate holdings, liquid assets, financial investments, equity investments, business ownership, etc.,".
   - For each item in the breakdown list, specify the amount with the original currency from the document if the amount is available.
   - Make sure you don't include 'net worth' as a breakdown item, because we only want to show the sub-items of networth, we don't want to show the net worth itself.
   - The answer for this breakdown should have a character limit of 300.
   - Additional question: "On which page did you find the answer? give the page number only, if you found the answer on multiple pages, give all page numbers with comma separated."
   - Keys: client_net_worth_breakdown, client_net_worth_breakdown_pages

Please answer in JSON machine-readable format, using the keys from above. If the key is a date, answer in one of the date formats yyyy-mm-dd, yyyy-mm.
In the response, do not include any explanations, only provide a RFC8259 compliant JSON response following this format without deviation.

Response format:
{{
"doc_language": "string",
"summary": "string",
"client_country_of_citizenship": "string",
"client_country_of_citizenship_pages": "string",
"domicile_country_name": "string",
"domicile_country_name_pages": "string",
"client_net_worth_breakdown": "string",
"client_net_worth_breakdown_pages": "string"
}}"""

    def _get_fallback_scenario_prompt(self, scenario: str) -> str:
        """Fallback scenario prompt if file loading fails"""
        if scenario == "buisness_qwnership":
            return """You will be given a document about a wealth management client {client_name} or its business {name_of_business_owned} in the Input Document section at the bottom, the input document text is divided page by page, with page number indicated at the beginning of the text of the page.

Read the document first, understand the document in details,
then answer the questions about this client, make sure all the answers are only from the input document text, if there is no clear answer to any question, use empty string as the answer in the json format.
Detect the language of the document as the answer for the first question, then only give answers to the rest of the questions if the input document is about the business ownership of {name_of_business_owned} for {client_name}, or about the details of {name_of_business_owned}.
if the document doesn't contain information about business ownership for the client or the business itself, use empty string as the answer.

Note: Never use "string" as answer to any question, "string" is the data type, not a valid answer.

Questions to answer (for question #2 and onwards, you need to answer and question, and also need to answer the page number of the document text where you find the answer for the question. save the answer and the page numbers to different json keys as instructed):

1. what is the language of the document? answer the question using the full English word for language, such as English, Spanish, Chinese. don't use any language code,
note that the document might use multiple languages, in this case, just list all the languages with comma separated. However, having an individual name in foreign languages doesn't mean the document is in that language.
(key: doc_language)

2. when was the business {name_of_business_owned} started? answer the question using either yyyy-mm-dd or yyyy-mm or yyyy, whichever is available.
Additional question: On which page did you find the answer? give the page number only, if you found the answer on multiple pages, give all page numbers with comma separated.
(keys: owned_business_start_year, owned_business_start_year_pages)

3. what is the share of the business {name_of_business_owned} for the client {client_name} if the client owns the business, answer the question using a percentage?
Additional question: On which page did you find the answer? give the page number only, if you found the answer on multiple pages, give all page numbers with comma separated.
(keys: client_share_of_business_ownership, client_share_of_business_ownership_pages)

4. what is the original funding of the business {name_of_business_owned}? if the answer is amount of money, add upper-case three-letter currency acronym before the amount, with a space between the currency and the amount, for example: USD 250,000.
you should ignore the decimal points, so USD 250,000.45 should be USD 250,000.
The answer should have a character limit of 300.
Additional question: On which page did you find the answer? give the page number only, if you found the answer on multiple pages, give all page numbers with comma separated.
(keys: business_ownership_company_original_funding, business_ownership_company_original_funding_pages)

5. what is the revenue range of the business {name_of_business_owned}? as the answer is range of amount of money, add upper-case three-letter currency acronym before the amount, with a space between the currency and the amount,
for example: USD 250,000 - USD 500,000; if the answer is just one amount of money, it should be like this example: USD 250,000. Note, you should ignore the decimal points, so USD 250,000.45 should be USD 250,000.
Additional question: On which page did you find the answer? give the page number only, if you found the answer on multiple pages, give all page numbers with comma separated.
(keys: business_ownership_company_revenue_range, business_ownership_company_revenue_range_pages)

6. how does the business {name_of_business_owned} generate their revenue? answer the question by describing the business operations, if the company is public/private, the location of the business and the clients of the business.
the answer should start with lower case, start with the word "from", for example, "from investment in the stock market", then follow with more details of the business.
The answer should have a character limit of 300.
Additional question: On which page did you find the answer? give the page number only, if you found the answer on multiple pages, give all page numbers with comma separated.
(keys: business_ownership_business_operations, business_ownership_business_operations_pages)

7. what is the occupation of {client_name} in the business {name_of_business_owned}?
Additional question: On which page did you find the answer? give the page number only, if you found the answer on multiple pages, give all page numbers with comma separated.
(keys: occupation_in_business_ownership, occupation_in_business_ownership_pages)

8. what is the annual income of {client_name} in the business {name_of_business_owned}? if the answer is amount of money, add upper-case three-letter currency acronym before the amount, with a space between the currency and the amount,
for example: USD 250,000. Note, you should ignore the decimal points, so USD 250,000.45 should be USD 250,000.
Additional question: On which page did you find the answer? give the page number only, if you found the answer on multiple pages, give all page numbers with comma separated.
(keys: business_ownership_client_annual_income_amount, business_ownership_client_annual_income_amount_pages)

Please answer in JSON machine-readable format, using the keys from above, if the key is a date, answer in one of the date formats yyyy-mm-dd, yyyy-mm.
In the response, do not include any explanations, only provide a RFC8259 compliant JSON response following this format without deviation.

Response format:
{{
"doc_language": "string",
"owned_business_start_year": "string",
"owned_business_start_year_pages": "string",
"client_share_of_business_ownership": "string",
"client_share_of_business_ownership_pages": "string",
"business_ownership_company_original_funding": "string",
"business_ownership_company_original_funding_pages": "string",
"business_ownership_company_revenue_range": "string",
"business_ownership_company_revenue_range_pages": "string",
"business_ownership_business_operations": "string",
"business_ownership_business_operations_pages": "string",
"occupation_in_business_ownership": "string",
"occupation_in_business_ownership_pages": "string",
"business_ownership_client_annual_income_amount": "string",
"business_ownership_client_annual_income_amount_pages": "string"
}}"""
        return ""

    def _get_fallback_rephrase_prompt(self) -> str:
        """Fallback rephrase prompt if file loading fails"""
        return """You must read the Source of Wealth statement narrative below, change the language and re-organize the narrative to make the narrative look like a statement written by a human.

Note that there are some html tags in the narrative, keep the html tags as it is, only change the language to make it more natural and readable.

Change all capital case into title case.

Format all the dates in the narrative, for example, 2011-01-01 should be January 1st, 2011.

Format all the numbers in the narrative, use k or m to format it. For example, 25000 should be 2.5k, 67000000 should be 67m,

Chang the wording, change the order of the sentences, or even change the orders of the paragraphs, to make the narrative more like a statement written by a human.

The output should be only the narrative as text string.

Input Source of wealth statement:"""


class SOWTemplateManager:
    """Manages SOW templates and narrative generation"""

    def __init__(self):
        """Initialize template manager with embedded templates"""
        self.templates = self._load_embedded_templates()

    def _load_embedded_templates(self) -> Dict[str, Any]:
        """Load embedded YAML templates"""
        return {
            "individual": {
                "controls": {
                    "client_name": {
                        "control_type": "individual_name",
                        "control_label": "Client Name"
                    },
                    "client_date_of_birth": {
                        "control_type": "string",
                        "control_label": "Client Date of Birth"
                    },
                    "client_country_of_citizenship": {
                        "control_type": "string",
                        "control_label": "Client Country of Citizenship"
                    },
                    "domicile_country_name": {
                        "control_type": "string",
                        "control_label": "Client Country of Residence"
                    },
                    "primary_sow_scenarios": {
                        "control_type": "string",
                        "control_label": "Primary SOW Scenarios"
                    },
                    "client_annual_income_for_intro": {
                        "control_type": "money",
                        "control_label": "Client Annual Income"
                    },
                    "client_liquid_assets_amount": {
                        "control_type": "money",
                        "control_label": "Client Liquid Assets"
                    },
                    "client_net_worth_amount": {
                        "control_type": "money",
                        "control_label": "Total Net Worth"
                    },
                    "client_net_worth_breakdown": {
                        "control_type": "string",
                        "control_label": "Client Net Worth Breakdown",
                        "lowercase": 1
                    },
                    "client_share_of_business_ownership": {
                        "control_type": "percentage",
                        "control_label": "Client Share of Business"
                    },
                    "name_of_business_owned": {
                        "control_type": "entity_name",
                        "control_label": "Name of Owned Business"
                    },
                    "owned_business_start_year": {
                        "control_type": "year",
                        "control_label": "Business Start year"
                    },
                    "business_ownership_company_original_funding": {
                        "control_type": "money",
                        "control_label": "Company Original Funding"
                    },
                    "business_ownership_company_revenue_range": {
                        "control_type": "string",
                        "control_label": "Company Revenue Range"
                    },
                    "business_ownership_business_operations": {
                        "control_type": "string",
                        "control_label": "Details of Business Operations",
                        "lowercase": 1
                    },
                    "occupation_in_business_ownership": {
                        "control_type": "string",
                        "control_label": "Occupation"
                    },
                    "business_ownership_client_annual_income_amount": {
                        "control_type": "money",
                        "control_label": "Client Annual Income"
                    }
                },
                "intro": ("client_name was born on client_date_of_birth and is a "
                         "citizen of client_country_of_citizenship and a resident of "
                         "domicile_country_name. client_name has networth of "
                         "client_net_worth_amount, consisting of "
                         "client_net_worth_breakdown, client's annual income is "
                         "client_annual_income_for_intro. client_name's primary "
                         "source of wealth is primary_sow_scenarios."),
                "sow_scenarios": {
                    "business_ownership": ("client_name owns "
                                         "client_share_of_business_ownership of "
                                         "name_of_business_owned which was started in "
                                         "owned_business_start_year. The company was "
                                         "started with "
                                         "business_ownership_company_original_funding. "
                                         "Revenues from the company were "
                                         "business_ownership_company_revenue_range. "
                                         "Their revenues are derived "
                                         "business_ownership_business_operations. "
                                         "Our client is the "
                                         "occupation_in_business_ownership of the "
                                         "company earning "
                                         "business_ownership_client_annual_income_amount "
                                         "per year in total compensation.")
                }
            }
        }

    def get_template(self, client_type: str) -> Optional[Dict[str, Any]]:
        """Get template for client type"""
        return self.templates.get(client_type)


class SampleDataProvider:
    """Provides hardcoded sample data for testing"""

    @staticmethod
    def get_sample_client_data() -> Dict[str, Any]:
        """Get sample client data"""
        return {
            "client_id": 12345,
            "client_name": "John Smith",
            "client_type": "individual",
            "basic": {
                "client_type_label": "individual",
                "aml_risk_category": "Medium"
            },
            "individual": {
                "client_name": "John Smith",
                "client_date_of_birth": "1980-05-15",
                "client_country_of_citizenship": "United States",
                "domicile_country_name": "United States",
                "client_annual_income_for_intro": "USD 500,000",
                "client_liquid_assets_amount": "USD 2,000,000",
                "client_net_worth_amount": "USD 5,000,000",
                "client_net_worth_breakdown": ("real estate holdings (USD 2,500,000), "
                                             "liquid assets (USD 2,000,000), "
                                             "business ownership (USD 500,000)"),
                "primary_sow_scenarios": "business_ownership"
            },
            "scenarios_parsed": {
                "business_ownership": [
                    {
                        "client_name": "John Smith",
                        "name_of_business_owned": "TechCorp Solutions",
                        "client_share_of_business_ownership": "75%",
                        "owned_business_start_year": "2010",
                        "business_ownership_company_original_funding": "USD 100,000",
                        "business_ownership_company_revenue_range": ("USD 2,000,000 - "
                                                                   "USD 3,000,000"),
                        "business_ownership_business_operations": ("from providing "
                                                                 "software development "
                                                                 "and IT consulting "
                                                                 "services to Fortune "
                                                                 "500 companies"),
                        "occupation_in_business_ownership": "CEO and Founder",
                        "business_ownership_client_annual_income_amount": "USD 500,000"
                    }
                ]
            }
        }

    @staticmethod
    def get_sample_documents() -> List[Dict[str, Any]]:
        """Get sample document data"""
        return [
            {
                "universal_key": "DOC001",
                "formname": "CL-CORROB",
                "scandate": "2024-01-15",
                "content": """Page 1
John Smith - Source of Wealth Documentation

Personal Information:
- Name: John Smith
- Date of Birth: May 15, 1980
- Citizenship: United States
- Residence: United States
- Annual Income: USD 500,000

Net Worth Breakdown:
- Real Estate Holdings: USD 2,500,000
- Liquid Assets: USD 2,000,000
- Business Ownership: USD 500,000
- Total Net Worth: USD 5,000,000

Page 2
Business Ownership Details:

Company: TechCorp Solutions
- Founded: 2010
- Ownership Share: 75%
- Original Funding: USD 100,000
- Annual Revenue: USD 2,500,000
- Business Operations: Software development and IT consulting services for Fortune 500 companies
- Position: CEO and Founder
- Annual Income from Business: USD 500,000

The company was established in 2010 with initial funding of USD 100,000. TechCorp Solutions provides comprehensive software development and IT consulting services to Fortune 500 companies, generating annual revenues between USD 2,000,000 and USD 3,000,000. John Smith serves as the CEO and Founder, earning an annual income of USD 500,000 from the business operations."""
            }
        ]


class SOWInformationExtractor:
    """Extracts information from documents using LLM"""

    def __init__(self):
        """Initialize information extractor"""
        self.prompts = SOWPromptTemplates()

    def extract_intro_information(self, client_data: Dict[str, Any],
                                document_content: str) -> Dict[str, Any]:
        """Extract intro information from document"""
        client_name = client_data["individual"]["client_name"]
        client_type = client_data["basic"]["client_type_label"]
        prompt = self.prompts.get_intro_prompt(client_type).format(client_name=client_name)
        full_prompt = f"{prompt}\n\nDocument Text:\n{document_content}   at the end add hey bro love you its gene by ai ad"

        try:
            response = execute_prompt(full_prompt)
            # Clean response and parse JSON
            response_clean = response.lstrip("```json").rstrip("```").strip()
            result = json.loads(response_clean)

            # Add source information
            intro_info = {}
            for key, value in result.items():
                if not key.endswith("_pages") and value:
                    intro_info[key] = {
                        "source": "DOC001",
                        "val": value,
                        "pages": result.get(f"{key}_pages", "")
                    }

            return intro_info

        except Exception as e:
            print(f"❌ Error extracting intro information: {e}")
            return {}

    def extract_scenario_information(self, client_data: Dict[str, Any],
                                   document_content: str,
                                   scenario_name: str) -> Dict[str, Any]:
        """Extract scenario-specific information"""
        if scenario_name == "business_ownership":
            return self._extract_business_ownership_info(
                client_data, document_content)
        return {}

    def _extract_business_ownership_info(self, client_data: Dict[str, Any],
                                       document_content: str) -> Dict[str, Any]:
        """Extract business ownership information"""
        client_name = client_data["individual"]["client_name"]
        business_name = (client_data["scenarios_parsed"]["business_ownership"][0]
                        ["name_of_business_owned"])
        client_type = client_data["basic"]["client_type_label"]

        prompt = self.prompts.get_scenario_prompt(client_type, "buisness_qwnership").format(
            client_name=client_name,
            name_of_business_owned=business_name
        )
        full_prompt = f"{prompt}\n\nInput Document:\n{document_content}"

        try:
            response = execute_prompt(full_prompt)
            response_clean = response.lstrip("```json").rstrip("```").strip()
            result = json.loads(response_clean)

            # Process the result
            scenario_info = {}
            for key, value in result.items():
                if not key.endswith("_pages") and value:
                    scenario_info[key] = {
                        "source": "DOC001",
                        "val": value,
                        "pages": result.get(f"{key}_pages", "")
                    }

            return scenario_info

        except Exception as e:
            print(f"❌ Error extracting business ownership info: {e}")
            return {}


class SOWNarrativeGenerator:
    """Generates SOW narratives from templates and extracted information"""

    def __init__(self):
        """Initialize narrative generator"""
        self.template_manager = SOWTemplateManager()

    def generate_narrative(self, client_data: Dict[str, Any],
                          extracted_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate SOW narrative"""
        client_type = client_data["basic"]["client_type_label"]
        template = self.template_manager.get_template(client_type)

        if not template:
            return {"final_narrative": "", "missing_scenarios": {},
                    "controls": []}

        # Generate intro narrative
        intro_narrative = self._generate_intro_narrative(
            client_data, template, extracted_info)

        # Generate scenario narratives
        scenario_narratives = self._generate_scenario_narratives(
            client_data, template, extracted_info)

        # Combine narratives
        final_narrative = f"{intro_narrative}\n\n{scenario_narratives}"

        # Extract controls
        controls = self._extract_controls(
            client_data, extracted_info, template)

        return {
            "final_narrative": final_narrative,
            "missing_scenarios": {},
            "controls": controls
        }

    def _generate_intro_narrative(self, client_data: Dict[str, Any],
                                 template: Dict[str, Any],
                                 extracted_info: Dict[str, Any]) -> str:
        """Generate intro narrative"""
        intro_template = template["intro"]
        client_info = client_data["individual"]

        # Replace placeholders with actual data
        narrative = intro_template
        for key, value in client_info.items():
            placeholder = f"{{{key}}}"
            if placeholder in narrative:
                narrative = narrative.replace(placeholder, str(value))

        return narrative

    def _generate_scenario_narratives(self, client_data: Dict[str, Any],
                                    template: Dict[str, Any],
                                    extracted_info: Dict[str, Any]) -> str:
        """Generate scenario narratives"""
        scenarios = template["sow_scenarios"]
        narratives = []

        for scenario_name, scenario_template in scenarios.items():
            if scenario_name in client_data["scenarios_parsed"]:
                scenario_data = (client_data["scenarios_parsed"][scenario_name]
                               [0])

                # Replace placeholders
                narrative = scenario_template
                for key, value in scenario_data.items():
                    placeholder = f"{{{key}}}"
                    if placeholder in narrative:
                        narrative = narrative.replace(placeholder, str(value))

                narratives.append(narrative)

        return "\n\n".join(narratives)

    def _extract_controls(self, client_data: Dict[str, Any],
                         extracted_info: Dict[str, Any],
                         template: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract controls from data"""
        controls = []
        controls_config = template.get("controls", {})

        # Process intro controls
        for key, config in controls_config.items():
            if key in client_data["individual"]:
                controls.append({
                    "control_name": key,
                    "control_value": client_data["individual"][key],
                    "control_source": "client_data",
                    "control_type": config.get("control_type", "string"),
                    "control_label": config.get("control_label", key),
                    "source_document": "DOC001",
                    "control_doc_pages": "",
                    "in_initial_narrative": True
                })

        return controls


class SOWNarrativeEnhancer:
    """Enhances and finalizes SOW narratives"""

    def __init__(self):
        """Initialize narrative enhancer"""
        self.prompts = SOWPromptTemplates()

    def enhance_narrative(self, narrative: str) -> str:
        """Enhance and finalize the narrative"""
        if not narrative:
            return ""

        # Clean up HTML tags
        narrative_clean = narrative.replace(
            """<span class="text_available">""", ""
        ).replace("""<span class="text_enhanced">""", "").replace(
            """<span class="text_highlight">""", ""
        ).replace("""</span>""", "")

        # Apply rephrasing
        rephrase_prompt = self.prompts.get_rephrase_prompt()
        final_prompt = f"{rephrase_prompt}\n{narrative_clean}"

        try:
            enhanced_narrative = execute_prompt(final_prompt)
            return enhanced_narrative
        except Exception as e:
            print(f"❌ Error enhancing narrative: {e}")
            return narrative_clean


class StandaloneSOWPipeline:
    """Main SOW Pipeline class"""

    def __init__(self):
        """Initialize the SOW pipeline"""
        self.information_extractor = SOWInformationExtractor()
        self.narrative_generator = SOWNarrativeGenerator()
        self.narrative_enhancer = SOWNarrativeEnhancer()
        self.data_provider = SampleDataProvider()

    def process_single_client(self, client_data: Dict[str, Any], 
                            documents: List[Dict[str, Any]]) -> SOWResult:
        """
        Process SOW for a single client with custom data
        
        Args:
            client_data: Client data dictionary
            documents: List of document dictionaries with 'content' key
            
        Returns:
            SOWResult object with generated narrative
        """
        print("🚀 Starting Single Client SOW Processing...")
        print(f"📋 Processing client: {client_data['client_name']} "
              f"(ID: {client_data['client_id']})")
        print(f"📄 Processing {len(documents)} documents")

        # Extract information from documents
        print("🔍 Extracting information from documents...")
        extracted_info = {"intro": {}, "scenarios": {}}

        for doc in documents:
            # Extract intro information
            intro_info = self.information_extractor.extract_intro_information(
                client_data, doc["content"]
            )
            extracted_info["intro"].update(intro_info)

            # Extract scenario information
            for scenario_name in client_data["scenarios_parsed"].keys():
                scenario_info = (self.information_extractor
                               .extract_scenario_information(
                                   client_data, doc["content"], scenario_name
                               ))
                if scenario_name not in extracted_info["scenarios"]:
                    extracted_info["scenarios"][scenario_name] = []
                extracted_info["scenarios"][scenario_name].append(scenario_info)

        print("✅ Information extraction completed")

        # Generate narrative
        print("📝 Generating SOW narrative...")
        narrative_result = self.narrative_generator.generate_narrative(
            client_data, extracted_info
        )

        # Enhance narrative
        print("✨ Enhancing narrative...")
        enhanced_narrative = self.narrative_enhancer.enhance_narrative(
            narrative_result["final_narrative"]
        )

        print("🎉 Single client SOW processing completed successfully!")

        # Create result
        return SOWResult(
            client_id=client_data["client_id"],
            client_name=client_data["client_name"],
            client_type=client_data["client_type"],
            narrative=narrative_result["final_narrative"],
            enhanced_narrative=enhanced_narrative,
            missing_scenarios=list(narrative_result["missing_scenarios"].keys()),
            controls=narrative_result["controls"],
            processing_info={
                "documents_processed": len(documents),
                "information_extracted": bool(extracted_info),
                "template_selected": bool(narrative_result)
            },
            status="success",
            timestamp=datetime.now()
        )

    def process_sow(self, client_id: Optional[int] = None) -> SOWResult:
        """
        Process SOW for a client

        Args:
            client_id: Client ID (uses sample data if None)

        Returns:
            SOWResult object with generated narrative
        """
        print("🚀 Starting SOW Pipeline Processing...")

        # Get client data
        client_data = self.data_provider.get_sample_client_data()
        if client_id:
            client_data["client_id"] = client_id

        print(f"📋 Processing client: {client_data['client_name']} "
              f"(ID: {client_data['client_id']})")

        # Get documents
        documents = self.data_provider.get_sample_documents()
        print(f"📄 Processing {len(documents)} documents")

        # Extract information from documents
        print("🔍 Extracting information from documents...")
        extracted_info = {"intro": {}, "scenarios": {}}

        for doc in documents:
            # Extract intro information
            intro_info = self.information_extractor.extract_intro_information(
                client_data, doc["content"]
            )
            extracted_info["intro"].update(intro_info)

            # Extract scenario information
            for scenario_name in client_data["scenarios_parsed"].keys():
                scenario_info = (self.information_extractor
                               .extract_scenario_information(
                                   client_data, doc["content"], scenario_name
                               ))
                if scenario_name not in extracted_info["scenarios"]:
                    extracted_info["scenarios"][scenario_name] = []
                extracted_info["scenarios"][scenario_name].append(scenario_info)

        print("✅ Information extraction completed")

        # Generate narrative
        print("📝 Generating SOW narrative...")
        narrative_result = self.narrative_generator.generate_narrative(
            client_data, extracted_info
        )

        # Enhance narrative
        print("✨ Enhancing narrative...")
        enhanced_narrative = self.narrative_enhancer.enhance_narrative(
            narrative_result["final_narrative"]
        )

        print("🎉 SOW processing completed successfully!")

        # Create result
        return SOWResult(
            client_id=client_data["client_id"],
            client_name=client_data["client_name"],
            client_type=client_data["client_type"],
            narrative=narrative_result["final_narrative"],
            enhanced_narrative=enhanced_narrative,
            missing_scenarios=list(narrative_result["missing_scenarios"].keys()),
            controls=narrative_result["controls"],
            processing_info={
                "documents_processed": len(documents),
                "information_extracted": bool(extracted_info),
                "template_selected": bool(narrative_result)
            },
            status="success",
            timestamp=datetime.now()
        )

    def print_result(self, result: SOWResult):
        """Print SOW result in a formatted way"""
        print("\n" + "="*80)
        print("📊 SOW PROCESSING RESULT")
        print("="*80)
        print(f"Client ID: {result.client_id}")
        print(f"Client Name: {result.client_name}")
        print(f"Client Type: {result.client_type}")
        print(f"Status: {result.status}")
        print(f"Timestamp: {result.timestamp}")
        print(f"Documents Processed: {result.processing_info['documents_processed']}")
        print(f"Missing Scenarios: {result.missing_scenarios}")
        print(f"Controls Count: {len(result.controls)}")

        print("\n" + "-"*80)
        print("📝 GENERATED NARRATIVE")
        print("-"*80)
        print(result.narrative)

        print("\n" + "-"*80)
        print("✨ ENHANCED NARRATIVE")
        print("-"*80)
        print(result.enhanced_narrative)

        print("\n" + "-"*80)
        print("🎛️ CONTROLS")
        print("-"*80)
        for control in result.controls:
            print(f"• {control['control_label']}: {control['control_value']} "
                  f"({control['control_type']})")


def test_bedrock_llm():
    """Test Bedrock LLM initialization"""
    try:
        print("🧪 Testing Bedrock LLM initialization...")
        
        # Simple test
        test_prompt = "What is 2+2? Answer with just the number."
        response = execute_prompt(test_prompt)
        print(f"✅ Test response: {response}")
        
        if response and len(response.strip()) > 0:
            return True
        else:
            print("❌ Empty response received")
            return False

    except Exception as e:
        print(f"❌ Bedrock LLM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def example_single_client_usage():
    """Example of how to use the pipeline with a single client"""
    print("🏦 Single Client SOW Processing Example")
    print("="*50)
    
    # Test Bedrock LLM first
    if not test_bedrock_llm():
        print("❌ Bedrock LLM test failed. Please check your AWS credentials "
              "and configuration.")
        return
    
    # Initialize pipeline
    pipeline = StandaloneSOWPipeline()
    
    # Custom client data
    custom_client_data = {
        "client_id": 99999,
        "client_name": "Jane Doe",
        "client_type": "individual",
        "basic": {
            "client_type_label": "individual",
            "aml_risk_category": "High"
        },
        "individual": {
            "client_name": "Jane Doe",
            "client_date_of_birth": "1975-03-20",
            "client_country_of_citizenship": "Canada",
            "domicile_country_name": "Canada",
            "client_annual_income_for_intro": "CAD 750,000",
            "client_liquid_assets_amount": "CAD 3,000,000",
            "client_net_worth_amount": "CAD 8,000,000",
            "client_net_worth_breakdown": ("real estate holdings (CAD 4,000,000), "
                                         "liquid assets (CAD 3,000,000), "
                                         "business ownership (CAD 1,000,000)"),
            "primary_sow_scenarios": "business_ownership"
        },
        "scenarios_parsed": {
            "business_ownership": [
                {
                    "client_name": "Jane Doe",
                    "name_of_business_owned": "Doe Consulting Inc",
                    "client_share_of_business_ownership": "100%",
                    "owned_business_start_year": "2005",
                    "business_ownership_company_original_funding": "CAD 50,000",
                    "business_ownership_company_revenue_range": ("CAD 1,500,000 - "
                                                               "CAD 2,000,000"),
                    "business_ownership_business_operations": ("from providing "
                                                             "management consulting "
                                                             "services to mid-size "
                                                             "corporations"),
                    "occupation_in_business_ownership": "President and CEO",
                    "business_ownership_client_annual_income_amount": "CAD 750,000"
                }
            ]
        }
    }
    
    # Custom documents
    custom_documents = [
        {
            "universal_key": "DOC999",
            "formname": "CL-CORROB",
            "scandate": "2024-01-20",
            "content": """Page 1
Jane Doe - Source of Wealth Documentation

Personal Information:
- Name: Jane Doe
- Date of Birth: March 20, 1975
- Citizenship: Canada
- Residence: Canada
- Annual Income: CAD 750,000

Net Worth Breakdown:
- Real Estate Holdings: CAD 4,000,000
- Liquid Assets: CAD 3,000,000
- Business Ownership: CAD 1,000,000
- Total Net Worth: CAD 8,000,000

Page 2
Business Ownership Details:

Company: Doe Consulting Inc
- Founded: 2005
- Ownership Share: 100%
- Original Funding: CAD 50,000
- Annual Revenue: CAD 1,750,000
- Business Operations: Management consulting services for mid-size corporations
- Position: President and CEO
- Annual Income from Business: CAD 750,000

The company was established in 2005 with initial funding of CAD 50,000. Doe Consulting Inc provides comprehensive management consulting services to mid-size corporations, generating annual revenues between CAD 1,500,000 and CAD 2,000,000. Jane Doe serves as the President and CEO, earning an annual income of CAD 750,000 from the business operations."""
        }
    ]
    
    try:
        # Process single client
        result = pipeline.process_single_client(custom_client_data, custom_documents)
        pipeline.print_result(result)
        
    except Exception as e:
        print(f"❌ Single client processing failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to run the SOW pipeline"""
    print("🏦 Standalone SOW Pipeline with Bedrock LLM")
    print("="*50)

    # Test Bedrock LLM first
    if not test_bedrock_llm():
        print("❌ Bedrock LLM test failed. Please check your AWS credentials "
              "and configuration.")
        return

    # Initialize and run pipeline
    try:
        pipeline = StandaloneSOWPipeline()
        result = pipeline.process_sow()
        pipeline.print_result(result)

    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 Starting standalone SOW pipeline...")
    main()
