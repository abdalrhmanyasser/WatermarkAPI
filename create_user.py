import secrets
from database import SessionLocal, init_db
from models import User

def create_api_user(username: str):
    init_db()
    db = SessionLocal()
    
    existing_user = db.query(User).filter(User.username == username).first()
    if existing_user:
        print(f"User '{username}' already exists. API Key: {existing_user.api_key}")
        db.close()
        return

    # Generate a secure 32-character API key
    api_key = secrets.token_urlsafe(32)
    
    new_user = User(username=username, api_key=api_key)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    print("\n--- USER CREATED SUCCESSFULLY ---")
    print(f"Username: {new_user.username}")
    print(f"API Key:  {new_user.api_key}")
    print("---------------------------------\n")
    print("Use this key in the 'X-API-Key' header of your requests.")
    
    db.close()

if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "test_user"
    create_api_user(name)