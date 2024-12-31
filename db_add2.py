import sqlite3
from datetime import datetime, timedelta
import random

# Connect to the database
conn = sqlite3.connect('appointus.db')
cursor = conn.cursor()

# Generate data covering all days of the week
days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
current_date = datetime.now()

# Add more queries
days_to_queries = {
    day: f"Help needed for {random.choice(['Plumbing', 'Electrical', 'Hair Styling', 'Internet']).lower()} services." 
    for day in days_of_week
}

for day, query_text in days_to_queries.items():
    query_date = current_date - timedelta(days=current_date.weekday() - days_of_week.index(day))
    seeker_id = random.randint(1, 7)
    is_converted = random.choice([0, 1])

    cursor.execute('''
    INSERT INTO queries (seeker_id, query_text, query_date, is_converted)
    VALUES (?, ?, ?, ?)
    ''', (seeker_id, query_text, query_date, is_converted))

# Add more appointments
days_to_appointments = {
    day: {
        "provider_id": random.randint(1, 4),
        "seeker_id": random.randint(1, 7),
        "service_type": random.choice(["Plumbing", "Electrical", "Hair Styling", "Internet"]),
        "appointment_date": current_date - timedelta(days=current_date.weekday() - days_of_week.index(day)),
        "status": random.choice(["Scheduled", "Completed", "Cancelled"]),
        "payment_amount": round(random.uniform(100, 1000), 2),
        "transaction_id": f"TXN{random.randint(100000, 999999)}"
    }
    for day in days_of_week
}

for appointment in days_to_appointments.values():
    cursor.execute('''
    INSERT INTO appointments (provider_id, seeker_id, service_type, appointment_date, status, payment_amount, transaction_id)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (appointment["provider_id"], appointment["seeker_id"], appointment["service_type"], 
          appointment["appointment_date"], appointment["status"], appointment["payment_amount"], 
          appointment["transaction_id"]))

# Add more feedback
days_to_feedback = {
    day: {
        "appointment_id": random.randint(1, 30),
        "rating": round(random.uniform(1, 5), 1),
        "comments": random.choice([
            "Excellent service!", "Pretty good experience.", "Could have been better.",
            "Would not recommend.", "Highly professional.", "Okay service.", "Amazing job!"
        ])
    }
    for day in days_of_week
}

for feedback in days_to_feedback.values():
    cursor.execute('''
    INSERT INTO feedback (appointment_id, rating, comments)
    VALUES (?, ?, ?)
    ''', (feedback["appointment_id"], feedback["rating"], feedback["comments"]))

# Commit and close
conn.commit()
conn.close()

print("Additional data for all days of the week added successfully!")
