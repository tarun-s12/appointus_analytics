import sqlite3
from datetime import datetime, timedelta
import random

# Connect to the database
conn = sqlite3.connect('appointus.db')
cursor = conn.cursor()

# Populate service_seekers table
seeker_name = ["Alice", "Bob", "Carol", "David", "Ellijah", "Flora", "Graham"]
seeker_locations = ["Street 1, CityA", "Street 2, CityA", "Street3, CityA", "Street 4, CityA", "Street 5, CityA", "Street 6, CityA", "Street 7, CityA"]
seeker_latitude = [12.967952, 12.964916, 12.963373, 12.979074, 12.970739, 12.967652, 12.978967]
seeker_longitude = [77.603393, 77.588289, 77.600734, 77.600235, 77.596824, 77.590248, 77.602465]

for i in range(len(seeker_name)):
    cursor.execute('''
    INSERT INTO service_seekers (name, location, latitude, longitude)
    VALUES (?, ?, ?, ?)
    ''', (seeker_name[i], seeker_locations[i], seeker_latitude[i], seeker_longitude[i]))

# Populate service_providers table
provider_name = ["Fix Speed Plumbers", "Crown Electrical Works", "Rich Look Saloon", "Q-bit Internet Services"]
expertise = ["Plumbing", "Electrical", "Hair Styling", "Internet"]
rating = [4.5, 4.2, 3.8, 3.5]
provider_locations = ["Street 1, CityA", "Street 1, CityA", "Street 10, CityA", "Street 7, CityA"]
provider_latitude = [12.967952, 12.964916, 12.963373, 12.979074]
provider_longitude = [77.603393, 77.588289, 77.600734, 77.600235]
phone = ["9876543210", "9876541230", "9998887776", "9182736540"]

for i in range(len(provider_name)):
    cursor.execute('''
    INSERT INTO service_providers (name, expertise, rating, location, latitude, longitude, phone)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (provider_name[i], expertise[i], rating[i], provider_locations[i], provider_latitude[i], provider_longitude[i], phone[i]))

# Populate queries table
for _ in range(30):
    seeker_id = random.randint(1, len(seeker_name))
    query_text = f"Need help with {random.choice(expertise).lower()} services."
    query_date = datetime.now() - timedelta(days=random.randint(1, 30))
    is_converted = random.choice([0, 1])

    cursor.execute('''
    INSERT INTO queries (seeker_id, query_text, query_date, is_converted)
    VALUES (?, ?, ?, ?)
    ''', (seeker_id, query_text, query_date, is_converted))

# Populate appointments table
for _ in range(30):
    provider_id = random.randint(1, len(provider_name))
    seeker_id = random.randint(1, len(seeker_name))
    service_type = expertise[provider_id - 1]
    appointment_date = datetime.now() + timedelta(days=random.randint(1, 30))
    status = random.choice(["Scheduled", "Completed", "Cancelled"])
    payment_amount = round(random.uniform(100, 1000), 2)
    transaction_id = f"TXN{random.randint(100000, 999999)}"

    cursor.execute('''
    INSERT INTO appointments (provider_id, seeker_id, service_type, appointment_date, status, payment_amount, transaction_id)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (provider_id, seeker_id, service_type, appointment_date, status, payment_amount, transaction_id))

# Populate feedback table
for _ in range(30):
    appointment_id = random.randint(1, 30)
    rating = round(random.uniform(1, 5), 1)
    comments = random.choice([
        "Great service!", "Satisfactory experience.", "Not up to the mark.", "Will recommend to others.", "Average service.",
        "Highly professional.", "Could be better.", "Fantastic experience.", "Poor quality work.", "Very reliable."
    ])

    cursor.execute('''
    INSERT INTO feedback (appointment_id, rating, comments)
    VALUES (?, ?, ?)
    ''', (appointment_id, rating, comments))

# Commit and close
conn.commit()
conn.close()

print("Data populated successfully!")
