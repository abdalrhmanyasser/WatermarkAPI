import getpass
import secrets
from database import SessionLocal, init_db
from models import User
from auth import get_password_hash


def create_user(email: str, password: str):
    init_db()
    db = SessionLocal()

    existing_user = db.query(User).filter(User.email == email).first()
    if existing_user:
        print(f"User '{email}' already exists.")
        db.close()
        return

    hashed = get_password_hash(password)
    new_user = User(email=email, hashed_password=hashed)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    print("\n--- USER CREATED SUCCESSFULLY ---")
    print(f"Email: {new_user.email}")
    print(f"User ID:  {new_user.id}")
    print("---------------------------------\n")

    db.close()


if __name__ == "__main__":
    import sys
    email = sys.argv[1] if len(sys.argv) > 1 else "user@example.com"
    pwd = sys.argv[2] if len(sys.argv) > 2 else getpass.getpass("Password: ")
    create_user(email, pwd)