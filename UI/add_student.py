#!/usr/bin/env python3
"""
Script to help add new students to the Physics Assistant
"""

import bcrypt

def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def generate_student_entry(username, name, email, password, course="PHYS101"):
    """Generate a student entry for the auth.py file"""
    hashed_password = hash_password(password)
    
    entry = f"""                '{username}': {{
                    'name': '{name}',
                    'password': self._hash_password('{password}'),
                    'email': '{email}',
                    'role': 'student',
                    'course': '{course}'
                }},"""
    
    return entry

def main():
    print("=== Physics Assistant - Add Student Tool ===")
    print()
    
    # Get student information
    username = input("Enter student username (e.g., john_doe): ").strip()
    name = input("Enter student full name (e.g., John Doe): ").strip()
    email = input("Enter student email (e.g., john.doe@university.edu): ").strip()
    password = input("Enter password for student: ").strip()
    course = input("Enter course code (default: PHYS101): ").strip() or "PHYS101"
    
    print("\\n=== Generated Entry ===")
    print("Add this to the 'usernames' section in frontend/components/auth.py:")
    print()
    print(generate_student_entry(username, name, email, password, course))
    print()
    print("=== Student Credentials ===")
    print(f"Username: {username}")
    print(f"Password: {password}")
    print(f"Full Name: {name}")
    print(f"Email: {email}")
    print(f"Course: {course}")

if __name__ == "__main__":
    main()