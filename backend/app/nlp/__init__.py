from openai import OpenAI
from fastapi import HTTPException
import logging
from ..config import OPENAI_API_KEY
from ..banking import BankingOperations
from ..config import DB_CONNECTION_STRING
import re
from typing import Dict, List
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)

# GPT Configuration
GPT_OPTIONS = {
    "model": "gpt-4-turbo-preview",
    "temperature": 0.7,
    "max_tokens": 150
}

# System message for chat context
SYSTEM_MESSAGE = """You are Alice, a helpful AI banking assistant for Bank of Kigali. Follow these guidelines:

1. ALWAYS maintain context of the conversation
2. If a user asks about balance or transactions:
   - If no account number provided, ask for it politely
   - If account number provided, respond with the actual balance
   - If they provided an account number earlier, use it without asking again
3. NEVER ask "what would you like to do with this account" after someone provides their number
4. Keep responses concise but friendly
5. When showing balance:
   - Format amounts as "RWF X,XXX"
   - Include last 3 transactions if available
   - Mention transaction dates

Example good responses:
- "I'll help you check your balance. Could you please provide your account number?"
- "Your current balance for account 100125540478 is RWF 50,000. Recent transactions:
   - March 15: ATM Withdrawal (-RWF 10,000)
   - March 14: Salary Deposit (+RWF 100,000)
   - March 13: Mobile Money (-RWF 5,000)"

Example bad responses:
- "What would you like to do with this account?"
- "How may I assist you with your account today?"
- "Thank you for providing your account number. What information do you need?"
"""

class Conversation:
    def __init__(self):
        self.messages: List[Dict[str, str]] = []
        self.account_number: str = None
        self.last_interaction: datetime = datetime.now()
        self.last_intent: str = None
    
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self.last_interaction = datetime.now()
    
    def is_expired(self) -> bool:
        return (datetime.now() - self.last_interaction) > timedelta(minutes=30)
    
    def get_messages(self) -> List[Dict[str, str]]:
        return [{"role": "system", "content": SYSTEM_MESSAGE}] + self.messages

class NLPProcessor:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.conversations: Dict[str, Conversation] = {}
        self.banking = BankingOperations(DB_CONNECTION_STRING)
    
    def extract_account_number(self, text: str) -> str:
        """Extract account number from text if present"""
        # Look for account number patterns (including with dashes)
        patterns = [
            r'\b\d{12}\b',  # 12 digits
            r'\b\d{4}[-\s]?\d{3}[-\s]?\d{4}\b'  # 4-3-4 format
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                # Remove any dashes or spaces
                return re.sub(r'[-\s]', '', match.group(0))
        return None

    def get_or_create_conversation(self, session_id: str) -> Conversation:
        """Get existing conversation or create new one"""
        if session_id not in self.conversations or self.conversations[session_id].is_expired():
            self.conversations[session_id] = Conversation()
        return self.conversations[session_id]

    def detect_intent(self, text: str) -> str:
        """Detect the user's intent from their message"""
        balance_patterns = [
            r"balance",
            r"how much.*have",
            r"statement",
            r"transactions",
            r"mini.?statement"
        ]
        
        if any(re.search(pattern, text.lower()) for pattern in balance_patterns):
            return "balance_inquiry"
        return None

    def format_transactions(self, transactions: List[dict]) -> str:
        """Format transactions for response"""
        if not transactions:
            return ""
            
        result = "\n\nRecent transactions:"
        for tx in transactions:
            amount = tx['amount']
            symbol = "+" if amount >= 0 else "-"
            result += f"\n- {tx['date']}: {tx['description']} ({symbol}RWF {abs(amount):,.0f})"
        return result

    async def process_text(self, text: str, session_id: str = "default") -> str:
        """Process text with GPT and return the response."""
        try:
            logger.info(f"Processing text with GPT: {text}")
            
            conversation = self.get_or_create_conversation(session_id)
            current_intent = self.detect_intent(text)
            
            # Add user message to conversation
            conversation.add_message("user", text)
            
            # Extract account number if present
            account_number = self.extract_account_number(text)
            if account_number:
                conversation.account_number = account_number
            
            # Handle different conversation scenarios
            if current_intent == "balance_inquiry" or conversation.last_intent == "balance_inquiry":
                if conversation.account_number:
                    try:
                        # Get recent transactions
                        transactions = self.banking.get_recent_transactions(conversation.account_number)
                        latest_balance = transactions[0].balance if transactions else 0
                        
                        # Format response with actual balance and transactions
                        response = f"Your current balance for account {conversation.account_number} is RWF {latest_balance:,.0f}"
                        response += self.format_transactions([tx.to_dict() for tx in transactions[:3]])
                        
                        conversation.add_message("assistant", response)
                        return response
                    except Exception as e:
                        logger.error(f"Banking operation error: {str(e)}")
                        response = "I apologize, but I'm having trouble accessing your account information right now. Please try again in a moment."
                        conversation.add_message("assistant", response)
                        return response
                else:
                    # Need account number
                    response = "I'll help you check your balance. Could you please provide your account number?"
                    conversation.add_message("assistant", response)
                    conversation.last_intent = "balance_inquiry"
                    return response
            
            # For other cases, use GPT for natural conversation
            completion = self.client.chat.completions.create(
                messages=conversation.get_messages(),
                **GPT_OPTIONS
            )
            
            response = completion.choices[0].message.content.strip()
            conversation.add_message("assistant", response)
            conversation.last_intent = current_intent
            
            logger.info(f"GPT response received: {response}")
            return response
            
        except Exception as e:
            logger.error(f"GPT processing error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"GPT processing failed: {str(e)}"
            )