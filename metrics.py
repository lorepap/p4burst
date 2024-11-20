import sqlite3
import time
import threading
import logging
from datetime import datetime
import os

class FlowMetricsManager:
    def __init__(self, exp_id=''):
        path = f"tmp/{exp_id}/"
        os.makedirs(path, exist_ok=True)
        self.db_path = os.path.join(path, f"flow_metrics.db")
        self.setup_database()

    # def initialize(self):
    #     """ Create a new database for flow metrics."""
    #     with sqlite3.connect(self.db_path) as conn:
    #         cursor = conn.cursor()
    #         cursor.execute('''CREATE TABLE IF NOT EXISTS flow_metrics (
    #                             flow_id INTEGER PRIMARY KEY,
    #                             sender_ip TEXT,
    #                             receiver_ip TEXT,
    #                             flow_size INTEGER,
    #                             start_time REAL,
    #                             end_time REAL,
    #                             fct REAL,
    #                             throughput_mbps REAL,
    #                             latency_ms REAL)''')
    #         # Table for query metrics
    #         cursor.execute('''CREATE TABLE IF NOT EXISTS query_metrics (
    #                             query_id INTEGER PRIMARY KEY,
    #                             start_time REAL,
    #                             end_time REAL,
    #                             qct REAL,
    #                             num_flows INTEGER,
    #                             avg_fct REAL,
    #                             total_throughput_mbps REAL)''')
    #     conn.commit()

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
                                latency_ms REAL,
                                type TEXT)''')
            # Table for query metrics
            cursor.execute('''CREATE TABLE IF NOT EXISTS query_metrics (
                                query_id INTEGER PRIMARY KEY,
                                start_time REAL,
                                end_time REAL,
                                qct REAL,
                                num_flows INTEGER,
                                avg_fct REAL,
                                total_throughput_mbps REAL,
                                type TEXT)''')
            conn.commit()

    def start_flow(self, flow_id, sender_ip, receiver_ip, flow_size, flow_type):
        """Record the start of a flow."""
        start_time = time.time()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''INSERT OR REPLACE INTO flow_metrics 
                              (flow_id, sender_ip, receiver_ip, flow_size, start_time, type)
                              VALUES (?, ?, ?, ?, ?)''', 
                           (flow_id, sender_ip, receiver_ip, flow_size, start_time, flow_type))
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
                throughput_mbps = (int(flow_size) * 8) / (float(fct) * 1e6)
                latency_ms = fct * 1000
                cursor.execute('''UPDATE flow_metrics 
                                  SET end_time = ?, fct = ?, throughput_mbps = ?, latency_ms = ?
                                  WHERE flow_id = ?''', 
                               (end_time, fct, throughput_mbps, latency_ms, flow_id))
                conn.commit()

    def complete_query(self, query_id, flow_ids):
        """Complete a query and save its metrics."""
        end_time = time.time()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Retrieve flow metrics for the query
            print(flow_ids)
            cursor.execute('''SELECT start_time, throughput_mbps, fct FROM flow_metrics 
                            WHERE flow_id IN ({})'''.format(','.join('?' * len(flow_ids))), flow_ids)
            flows = cursor.fetchall()

            if flows:
                min_start_time = min([flow[0] for flow in flows])
                print('min start time', min_start_time)
                qct = end_time - min_start_time
                print('qct', qct)
                avg_fct = sum(f[2] for f in flows) / len(flows)  # Average flow completion time
                total_throughput_mbps = sum(f[1] for f in flows)  # Sum of flow throughputs

                # Insert query metrics
                cursor.execute('''INSERT INTO query_metrics 
                                  (query_id, start_time, end_time, qct, num_flows, avg_fct, total_throughput_mbps) 
                                  VALUES (?, ?, ?, ?, ?, ?, ?)''', 
                               (query_id, min_start_time, end_time, qct, len(flows), avg_fct, total_throughput_mbps))
                conn.commit()

    def get_flow_metrics(self):
        """Retrieve all flow metrics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''SELECT * FROM flow_metrics''')
            return cursor.fetchall()

    def get_query_metrics(self):
        """Retrieve all query metrics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''SELECT * FROM query_metrics''')
            return cursor.fetchall()
