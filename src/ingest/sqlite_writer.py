import sqlite3
import os

def get_db_connection():
    """
    Returns a connection to the SQLite database.
    """
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data.db'))
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def create_tables():
    """
    Creates the necessary tables in the database.
    """
    conn = get_db_connection()
    c = conn.cursor()
    
    # Create raw_bars table
    c.execute('''
        CREATE TABLE IF NOT EXISTS raw_bars (
            symbol TEXT,
            timestamp INTEGER,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (symbol, timestamp)
        )
    ''')
    
    # Create raw_navs table
    c.execute('''
        CREATE TABLE IF NOT EXISTS raw_navs (
            symbol TEXT,
            date TEXT,
            nav REAL,
            PRIMARY KEY (symbol, date)
        )
    ''')
    
    # Create pelosi_trades table
    c.execute('''
        CREATE TABLE IF NOT EXISTS pelosi_trades (
            disclosure_year INTEGER,
            disclosure_date TEXT,
            transaction_date TEXT,
            owner TEXT,
            ticker TEXT,
            asset_description TEXT,
            type TEXT,
            amount TEXT,
            representative TEXT,
            district TEXT,
            ptr_link TEXT,
            cap_gains_over_200_usd BOOLEAN
        )
    ''')
    
    # Create features table
    c.execute('''
        CREATE TABLE IF NOT EXISTS features (
            symbol TEXT,
            timestamp INTEGER,
            return_5 REAL,
            return_15 REAL,
            return_60 REAL,
            volatility_20 REAL,
            volume_ratio_20 REAL,
            is_pelosi INTEGER,
            is_quantum INTEGER,
            is_dynamic INTEGER,
            category_weight REAL,
            PRIMARY KEY (symbol, timestamp)
        )
    ''')
    
    conn.commit()
    conn.close()

def write_raw_bars(bars):
    """
    Writes raw bar data to the database.
    """
    conn = get_db_connection()
    c = conn.cursor()
    c.executemany('INSERT OR REPLACE INTO raw_bars VALUES (?, ?, ?, ?, ?, ?, ?)', bars)
    conn.commit()
    conn.close()

def write_raw_navs(navs):
    """
    Writes raw NAV data to the database.
    """
    conn = get_db_connection()
    c = conn.cursor()
    c.executemany('INSERT OR REPLACE INTO raw_navs VALUES (?, ?, ?)', navs)
    conn.commit()
    conn.close()

def write_pelosi_trades(trades):
    """
    Writes Pelosi trade data to the database.
    """
    conn = get_db_connection()
    c = conn.cursor()
    c.executemany('INSERT OR REPLACE INTO pelosi_trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', trades)
    conn.commit()
    conn.close()

def write_features(features_df):
    """
    Writes the features to the database.
    """
    conn = get_db_connection()
    features_df.to_sql('features', conn, if_exists='replace', index=False)
    conn.close()

if __name__ == '__main__':
    create_tables()
    print("Database and tables created successfully.")
