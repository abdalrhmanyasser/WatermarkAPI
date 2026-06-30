import getpass
from database import get_db
from auth import get_password_hash

def create_user(full_name: str, email: str, password: str):
    db = next(get_db())

    existing_user = db.users.find_one({"email": email})
    if existing_user:
        print(f"User '{email}' already exists.")
        return

    user_doc = {
        "full_name": full_name,
        "email": email,
        "hashed_password": get_password_hash(password)
    }
    
    res = db.users.insert_one(user_doc)

    print("\n--- USER CREATED SUCCESSFULLY ---")
    print(f"Name:  {full_name}")
    print(f"Email: {email}")
    print(f"User ID:  {res.inserted_id}")
    print("---------------------------------\n")

if __name__ == "__main__":
    import sys
    full_name = sys.argv[1] if len(sys.argv) > 1 else input("Full Name: ")
    email = sys.argv[2] if len(sys.argv) > 2 else input("Email: ")
    pwd = sys.argv[3] if len(sys.argv) > 3 else getpass.getpass("Password: ")
    create_user(full_name, email, pwd)