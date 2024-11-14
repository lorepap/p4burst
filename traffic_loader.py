import sqlite3
from collections import defaultdict

class TrafficDataLoader:
    def __init__(self, db_paths):
        self.db_paths = db_paths
        self.flow_data = defaultdict(list)  # Each server column will have a list of flow attributes

    def load_data(self):
        # Connect to each database file
        conn_flow_ids = sqlite3.connect(self.db_paths['flow_ids'])
        conn_flow_size = sqlite3.connect(self.db_paths['flow_size'])
        conn_inter_arrival = sqlite3.connect(self.db_paths['inter_arrival_time'])
        conn_server_idx = sqlite3.connect(self.db_paths['server_idx'])

        # Retrieve data from each table
        cursor_flow_ids = conn_flow_ids.cursor()
        cursor_flow_size = conn_flow_size.cursor()
        cursor_inter_arrival = conn_inter_arrival.cursor()
        cursor_server_idx = conn_server_idx.cursor()

        cursor_flow_ids.execute("SELECT * FROM flow_ids")
        cursor_flow_size.execute("SELECT * FROM flow_size")
        cursor_inter_arrival.execute("SELECT * FROM inter_arrival_time")
        cursor_server_idx.execute("SELECT * FROM server_idx")

        flow_id_rows = cursor_flow_ids.fetchall()
        flow_size_rows = cursor_flow_size.fetchall()
        inter_arrival_rows = cursor_inter_arrival.fetchall()
        server_idx_rows = cursor_server_idx.fetchall()

        # Assuming columns like server0app1, server1app1, etc.
        column_names = [desc[0] for desc in cursor_flow_ids.description]

        for i, server in enumerate(column_names):
            for flow_id, flow_size, inter_arrival, destination in zip(
                flow_id_rows[i], flow_size_rows[i], inter_arrival_rows[i], server_idx_rows[i]
            ):
                self.flow_data[server].append({
                    'flow_id': flow_id,
                    'flow_size': flow_size,
                    'inter_arrival': inter_arrival,
                    'destination_ip': destination  # Resolve IP in run_background_app
                })

        # Close connections
        conn_flow_ids.close()
        conn_flow_size.close()
        conn_inter_arrival.close()
        conn_server_idx.close()

    def get_flow_data(self, server_name):
        return self.flow_data.get(server_name, [])
