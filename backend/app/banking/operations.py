import pyodbc
from fastapi import HTTPException
import logging
from typing import List
from datetime import datetime, timedelta
from .models import Transaction

logger = logging.getLogger(__name__)

class BankingOperations:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
    
    def _get_connection(self):
        try:
            return pyodbc.connect(self.connection_string)
        except Exception as e:
            logger.error(f"Database connection error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Failed to connect to the banking system"
            )
    
    def get_statement(
        self, 
        account_number: str, 
        start_date: str, 
        end_date: str
    ) -> List[Transaction]:
        """
        Retrieve account statement for the given period
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "EXEC PS_BK_Statement ?, ?, ?",
                    (account_number, start_date, end_date)
                )
                
                transactions = []
                for row in cursor.fetchall():
                    transactions.append(Transaction(
                        date=row[2],
                        description=row[5],
                        amount=float(row[7] - row[8]),  # Credit - Debit
                        balance=float(row[9]),
                        reference=row[1] if row[1] != '-' else None
                    ))
                
                return transactions
                
        except Exception as e:
            logger.error(f"Statement retrieval error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve account statement"
            )
    
    def get_recent_transactions(
        self, 
        account_number: str, 
        days: int = 3
    ) -> List[Transaction]:
        """
        Get recent transactions for the specified number of days
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        return self.get_statement(
            account_number,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )