from enum import Enum

DOMAIN = "garage_door_vision"

CONF_CAMERA = "camera_entity"
CONF_SCAN_INTERVAL = "scan_interval"
CONF_BUTTON = "button_entity"

DEFAULT_SCAN_INTERVAL = 5  # seconds

ATTR_STATE_CONFIDENCE = "confidence"
CONF_BUTTON = "button_entity"

STORAGE_VERSION = 1
STORAGE_KEY = "garage_door_vision_calibration"

CONF_TRACKER_URL = "tracker_url"


class PlannerState(Enum):
    OPEN = 1
    CLOSE = 2
    OPENING = 3
    CLOSING = 4
    UNKNOWN = 5
