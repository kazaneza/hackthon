from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class Transaction:
    date: datetime
    description: str
    amount: float
    balance: float
    reference: Optional[str] = None
    
    @property
    def formatted_date(self) -> str:
        return self.date.strftime('%Y-%m-%d')
    
    @property
    def is_debit(self) -> bool:
        return self.amount < 0
    
    @property
    def formatted_amount(self) -> str:
        return f"RWF {abs(self.amount):,.0f}"
    
    def to_dict(self):
        return {
            'date': self.formatted_date,
            'description': self.description,
            'amount': self.amount,
            'balance': self.balance,
            'reference': self.reference
        }