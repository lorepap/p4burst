import sqlite3
import time
import threading
import logging
from datetime import datetime

class FlowMetricsManager:
    def __init__(self, exp_id=0):
        self.db_path = f"flow_metrics.db"
        self.setup_database()

    def setup_database(self):
        """Setup SQLite database to store flow metrics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS flow_metrics (
                                flow_id INTEGER PRIMARY KEY,
                                sender_ip TEXT,
                                receiver_ip TEXT,
                                flow_size INTEGER,
                                start_time REAL,
                                end_time REAL,
                                fct REAL,
                                throughput_mbps REAL,
                                latency_ms REAL)''')
            conn.commit()

    def start_flow(self, flow_id, sender_ip, receiver_ip, flow_size):
        """Record the start of a flow."""
        start_time = time.time()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''INSERT OR REPLACE INTO flow_metrics 
                              (flow_id, sender_ip, receiver_ip, flow_size, start_time)
                              VALUES (?, ?, ?, ?, ?)''', 
                           (flow_id, sender_ip, receiver_ip, flow_size, start_time))
            conn.commit()

    def complete_flow(self, flow_id):
        """Complete the flow and record metrics."""
        end_time = time.time()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''SELECT start_time, flow_size FROM flow_metrics WHERE flow_id = ?''', (flow_id,))
            result = cursor.fetchone()
            if result:
                start_time, flow_size = result
                fct = end_time - start_time
                throughput_mbps = (flow_size * 8) / (fct * 1e6)
                latency_ms = fct * 1000
                cursor.execute('''UPDATE flow_metrics 
                                  SET end_time = ?, fct = ?, throughput_mbps = ?, latency_ms = ?
                                  WHERE flow_id = ?''', 
                               (end_time, fct, throughput_mbps, latency_ms, flow_id))
                conn.commit()

    def get_metrics(self):
        """Retrieve all flow metrics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''SELECT * FROM flow_metrics''')
            return cursor.fetchall()
