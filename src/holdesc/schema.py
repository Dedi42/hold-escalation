"""Standard (company-neutral) schema used by every module.

Raw MES exports differ between plants. A local, untracked YAML file maps the raw
column names to the names below (see config/schema.example.yaml). No plant-specific
column name, code, or identifier is hard-coded in this package.
"""
HOLD_COLUMNS = {
    "lot_id": "lot identifier",
    "step": "process step at which the hold was raised",
    "reason": "hold reason code",
    "department": "department responsible for the disposition",
    "start_time": "time the hold was raised",
    "release_time": "time the hold was released (empty if still open)",
    "priority": "lot priority (numeric)",
    "lot_type": "lot type, e.g. production or engineering",
    "product_family": "package or product family",
    "quantity": "number of wafers in the lot",
    "rework": "1 if the lot is in rework, else 0",
}
LOT_COLUMNS = {
    "lot_id": "lot identifier",
    "completion_time": "time the lot left the line",
    "due_time": "customer due date (shipping-out date)",
    "risk": "optional delivery-risk score in [0, 1]",
}
REQUIRED_HOLD = ["lot_id", "step", "reason", "department", "start_time", "release_time"]
