#!/usr/bin/env python3
"""
Generate the correct password hash for admin user
Run this script to get the updated password hash for 'admin123'
"""

from werkzeug.security import generate_password_hash, check_password_hash

# Your admin password
admin_password = "admin123"

# Generate new hash
new_hash = generate_password_hash(admin_password)

# Test the current hash from your database
current_hash = "pbkdf2:sha256:150000$0dsQUPzZ$a785632d0a1f6a8c9b2cac47dfa153f39e9af92d05f38db2cbf82e4c9fbb4749"

print("=" * 80)
print("51Talk Learning Platform - Admin Password Hash Verification")
print("=" * 80)
print(f"Password to hash: {admin_password}")
print()
print("TESTING CURRENT HASH:")
print(f"Current hash: {current_hash}")
print(f"Does current hash match 'admin123'? {check_password_hash(current_hash, admin_password)}")
print()
print("GENERATING NEW HASH:")
print(f"New hash: {new_hash}")
print(f"Does new hash match 'admin123'? {check_password_hash(new_hash, admin_password)}")
print()
print("=" * 80)
print("SQL COMMANDS TO FIX THE ISSUE:")
print("=" * 80)
print()
print("Option 1: Update existing admin user:")
print(f"UPDATE admin_users SET password = '{new_hash}' WHERE username = 'admin';")
print()
print("Option 2: Delete and recreate admin user:")
print("DELETE FROM admin_users WHERE username = 'admin';")
print(f"INSERT INTO admin_users (username, password) VALUES ('admin', '{new_hash}');")
print()
print("=" * 80)
print("INSTRUCTIONS:")
print("1. Connect to your PostgreSQL database")
print("2. Run one of the SQL commands above")
print("3. Try logging in again with username: admin, password: admin123")
print("=" * 80)