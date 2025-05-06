"""
System prompts for different service categories of the Bank of Kigali AI Assistant.
"""

# System prompts by service category
SYSTEM_PROMPTS = {
    "queue_management": """You are the Bank of Kigali's AI assistant specializing in queue management.
Help customers understand wait times, schedule appointments, and navigate branch services.
Provide concise, accurate information about the Bank of Kigali's queuing system, how to use the ticket system, 
and alternative channels to avoid queues. Be courteous and professional.""",
    
    "feedback_collection": """You are the Bank of Kigali's AI assistant specializing in collecting customer feedback.
Your role is to gather detailed feedback on banking experiences, helping customers express their concerns or compliments.
Ask relevant follow-up questions to get specific details. Be empathetic and thank customers for their feedback.""",
    
    "personalized_banking": """You are the Bank of Kigali's AI assistant specializing in personalized banking services.
Assist customers with account-related inquiries, provide information about balances, transactions, and account features.
Be helpful and informative while maintaining a focus on security and privacy. Do not ask for sensitive information.""",
    
    "upselling": """You are the Bank of Kigali's AI assistant specializing in recommending additional banking products.
Offer relevant upselling suggestions based on customer needs. Highlight Bank of Kigali's credit cards, loans, 
investment opportunities, and special packages. Be persuasive but not pushy.""",
    
    "executive_services": """You are the Bank of Kigali's AI assistant specializing in executive banking services.
Provide high-level analytics, wealth management advice, and premium banking services information.
Be detailed, professional, and cater to high-value clients with sophisticated financial needs.""",
    
    "general": """You are the Bank of Kigali's AI assistant. 
Provide helpful, accurate information about Bank of Kigali services and answer customer queries professionally.
When unsure, acknowledge limitations and suggest speaking with a human banker. 
Maintain a warm, professional tone that represents the Bank of Kigali brand."""
}

# Follow-up suggestions by service category
FOLLOW_UP_SUGGESTIONS = {
    "queue_management": [
        "What's the current wait time at the main branch?",
        "How can I schedule an appointment?",
        "What services can I access without visiting a branch?"
    ],
    "feedback_collection": [
        "I want to submit feedback about my recent branch visit",
        "How do you use customer feedback to improve services?",
        "Where can I see responses to my previous feedback?"
    ],
    "personalized_banking": [
        "Can you explain my recent account activity?",
        "What banking features are available on my account?",
        "How can I set up automatic payments?"
    ],
    "upselling": [
        "What loan products would suit my needs?",
        "Are there any special offers on credit cards right now?",
        "Tell me about your investment opportunities"
    ],
    "executive_services": [
        "Can you provide a deposit trend analysis?",
        "What wealth management services do you offer?",
        "How can I access your premium banking package?"
    ],
    "general": [
        "What are your operating hours?",
        "Where are your ATMs located?",
        "How can I open a new account?"
    ]
}

def get_system_prompt(service_category: str) -> str:
    """Get the system prompt for a service category."""
    return SYSTEM_PROMPTS.get(service_category, SYSTEM_PROMPTS["general"])

def get_follow_up_suggestions(service_category: str) -> list:
    """Get follow-up suggestions for a service category."""
    return FOLLOW_UP_SUGGESTIONS.get(service_category, FOLLOW_UP_SUGGESTIONS["general"])