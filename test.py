import uuid


def generate_session_id():
    return str(uuid.uuid4())


# Generate and print a session ID
session_id = generate_session_id()
print("Your session id is:", session_id)
