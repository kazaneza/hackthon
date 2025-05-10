"""
Test script for the enhanced message store functionality.
File: nlp/test_enhanced_memory.py
"""

from core.enhanced_message_store import EnhancedMessageStore, extract_user_information_from_qa
import time
import json

def test_enhanced_memory():
    """Test the enhanced message store functionality."""
    
    # Initialize the store
    store = EnhancedMessageStore(max_pairs=3, expiry_seconds=10)  # Short expiry for testing
    
    print("=== Testing Enhanced Message Store ===\n")
    
    # Test 1: Basic Q&A pair storage
    print("Test 1: Adding Q&A pairs")
    user_id = "test_user_1"
    
    # First interaction - user introduces themselves
    question1 = "Hi, my name is John Doe"
    answer1 = "Hello John Doe! Welcome to Bank of Kigali. How can I assist you today?"
    user_info1 = extract_user_information_from_qa(question1, answer1)
    print(f"Detected user info: {user_info1}")
    
    store.add_qa_pair(user_id, question1, answer1, user_info1)
    
    # Second interaction
    question2 = "What is my account balance?"
    answer2 = "I'll help you check your account balance. Could you please provide your account number?"
    store.add_qa_pair(user_id, question2, answer2)
    
    # Third interaction - provide account number
    question3 = "My account number is 1234567890"
    answer3 = "Thank you! I'm checking your account balance for account 1234567890."
    user_info3 = extract_user_information_from_qa(question3, answer3)
    print(f"Detected account info: {user_info3}")
    
    store.add_qa_pair(user_id, question3, answer3, user_info3)
    
    # Test 2: Retrieve stored information
    print("\nTest 2: Retrieving stored conversations")
    conversations = store.get_recent_conversations(user_id)
    print(f"Number of stored conversations: {len(conversations)}")
    
    for i, pair in enumerate(conversations, 1):
        print(f"\nConversation {i}:")
        print(f"  Question: {pair.question}")
        print(f"  Answer: {pair.answer}")
        print(f"  User info extracted: {pair.user_info}")
    
    user_info = store.get_user_info(user_id)
    print(f"\nPersistent user info: {user_info}")
    
    # Test 3: Test max_pairs limit
    print("\nTest 3: Testing max_pairs limit (3)")
    # Add a 4th conversation to test limit
    question4 = "What are your interest rates?"
    answer4 = "Our interest rates vary by product type. What specific product are you interested in?"
    store.add_qa_pair(user_id, question4, answer4)
    
    conversations = store.get_recent_conversations(user_id)
    print(f"After adding 4th conversation, stored pairs: {len(conversations)}")
    print("First conversation should be removed:")
    for i, pair in enumerate(conversations, 1):
        print(f"  {i}. {pair.question[:30]}...")
    
    # Test 4: Test memory with name retrieval
    print("\nTest 4: Testing name retrieval scenario")
    question5 = "What is my name?"
    answer5 = f"Your name is {user_info.get('name', 'John Doe')}."
    store.add_qa_pair(user_id, question5, answer5)
    
    # Create context for AI
    def create_test_context(user_id):
        user_info = store.get_user_info(user_id)
        conversations = store.get_recent_conversations(user_id)
        
        context_parts = []
        
        if user_info:
            context_parts.append("=== USER INFORMATION ===")
            for key, value in user_info.items():
                context_parts.append(f"{key}: {value}")
            context_parts.append("")
        
        if conversations:
            context_parts.append("=== RECENT CONVERSATION HISTORY ===")
            for i, pair in enumerate(conversations, 1):
                context_parts.append(f"[Q{i}] User: {pair.question}")
                context_parts.append(f"[A{i}] Assistant: {pair.answer}")
                context_parts.append("")
        
        return "\n".join(context_parts)
    
    context = create_test_context(user_id)
    print("Generated context for AI:")
    print("-" * 50)
    print(context)
    print("-" * 50)
    
    # Test 5: Test multiple users
    print("\nTest 5: Testing multiple users")
    user_id2 = "test_user_2"
    store.add_qa_pair(user_id2, "Hi, I'm Jane Smith", "Hello Jane! How can I help you?", 
                      {"name": "Jane Smith"})
    
    print(f"User 1 conversations: {len(store.get_recent_conversations(user_id))}")
    print(f"User 2 conversations: {len(store.get_recent_conversations(user_id2))}")
    print(f"User 1 info: {store.get_user_info(user_id)}")
    print(f"User 2 info: {store.get_user_info(user_id2)}")
    
    # Test 6: Test expiry (if you want to test this, increase sleep time)
    print("\nTest 6: Testing expiry functionality")
    print("Waiting for expiry... (this will take 10 seconds)")
    time.sleep(11)  # Wait for expiry
    
    # Check if conversations expired
    conversations_after_expiry = store.get_recent_conversations(user_id)
    user_info_after_expiry = store.get_user_info(user_id)
    
    print(f"Conversations after expiry: {len(conversations_after_expiry)}")
    print(f"User info after expiry: {user_info_after_expiry}")
    print("Note: User info persists even after conversation expiry")
    
    # Test 7: Test clear data
    print("\nTest 7: Testing clear user data")
    store.clear_user_data(user_id)
    print(f"After clearing - Conversations: {len(store.get_recent_conversations(user_id))}")
    print(f"After clearing - User info: {store.get_user_info(user_id)}")
    
    print("\n=== All tests completed ===")

if __name__ == "__main__":
    test_enhanced_memory()