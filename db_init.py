import sqlite3

conn = sqlite3.connect('appointus.db')

cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS service_providers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    expertise TEXT,
    rating REAL,
    location TEXT,
    latitude REAL,
    longitude REAL,
    phone STRING CHECK(LENGTH(phone) = 10)
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS service_seekers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    location STRING,
    latitude REAL,
    longitude REAL
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    seeker_id INTEGER,
    query_text TEXT NOT NULL,
    query_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_converted BOOLEAN DEFAULT 0,
    FOREIGN KEY (seeker_id) REFERENCES service_seekers (id)
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS appointments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_id INTEGER,
    seeker_id INTEGER,
    service_type TEXT NOT NULL,
    appointment_date TIMESTAMP,
    status TEXT DEFAULT 'Scheduled',
    payment_amount REAL,
    transaction_id TEXT,
    FOREIGN KEY (provider_id) REFERENCES service_providers (id),
    FOREIGN KEY (seeker_id) REFERENCES service_seekers (id)
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    appointment_id INTEGER,
    rating REAL NOT NULL,
    comments TEXT,
    FOREIGN KEY (appointment_id) REFERENCES appointments (id)
)
''')

conn.commit()
conn.close()

print("Database and tables created successfully!")
