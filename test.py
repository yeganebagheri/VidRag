import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Supabase database connection details
SUPABASE_URL = os.getenv('SUPABASE_URL')  # Your Supabase project URL
SUPABASE_PASSWORD = os.getenv('SUPABASE_PASSWORD')  # Your database password
DATABASE_NAME = os.getenv('DATABASE_NAME', 'postgres')  # Usually 'postgres'

# Create connection string
# Format: postgresql://postgres:[password]@[host]:[port]/[database]
connection_string = os.getenv('DATABASE_URL')
# Create engine
engine = create_engine(connection_string)

# Create session
Session = sessionmaker(bind=engine)

def test_connection():
    """Test the database connection"""
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            print("✅ Database connection successful!")
            return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def get_session():
    """Get a new database session"""
    return Session()

if name == "__main__":
    # Test the connection
    test_connection()
    
    # Example usage
    session = get_session()
    try:
        # Example query - get all table names
        result = session.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"))
        tables = result.fetchall()
        print(f"📋 Tables in database: {[table[0] for table in tables]}")
    except Exception as e:
        print(f"Error querying database: {e}")
    finally:
        session.close()