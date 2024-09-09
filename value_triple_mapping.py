# value_triple_mapping.py

class ValueTripleMapping:
    def __init__(self):
        self.mapping = {
            "op-reachability": ["Min op-reachability (%)", "Max op-reachability (%)", "Avg op-reachability (%)"],
            "in-octets": ["Min in-octets (kbit/s)", "Max in-octets (kbit/s)", "Avg in-octets (kbit/s)"],
            "in-utilization": ["Min in-utilization (%)", "Max in-utilization (%)", "Avg in-utilization (%)"],
            "in-errors": ["Min in-errors (%)", "Max in-errors (%)", "Avg in-errors (%)"],
            "in-discards": ["Min in-discards (%)", "Max in-discards (%)", "Avg in-discards (%)"],
            "out-octets": ["Min out-octets (kbit/s)", "Max out-octets (kbit/s)", "Avg out-octets (kbit/s)"],
            "out-utilization": ["Min out-utilization (%)", "Max out-utilization (%)", "Avg out-utilization (%)"],
            "out-errors": ["Min out-errors (%)", "Max out-errors (%)", "Avg out-errors (%)"],
            "out-discards": ["Min out-discards (%)", "Max out-discards (%)", "Avg out-discards (%)"]
        }

    def get_mapping(self):
        return self.mapping