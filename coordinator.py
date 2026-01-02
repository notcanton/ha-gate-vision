from __future__ import annotations

from datetime import datetime
import logging

import cv2
import numpy as np

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator

from .camera_reader import get_camera_frame
from .const import CONF_CAMERA, STORAGE_KEY, STORAGE_VERSION, PlannerState

_LOGGER = logging.getLogger(__name__)


class GarageDoorTracker:
    def __init__(self):
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        self.id = 1

    def pos(self, frame: bytes | None) -> np.ndarray | None:
        if frame is None:
            return None

        nparr = np.frombuffer(frame, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_np = cv2.resize(img_np, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

        import time

        start_time = time.perf_counter()
        corners, ids, _ = self.detector.detectMarkers(gray)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        _LOGGER.info(f"Execution time: {elapsed_time:.4f} seconds")

        if ids is not None:
            _LOGGER.info("Detected THINGY")
            cv2.aruco.drawDetectedMarkers(img_np, corners, ids)
            success = cv2.imwrite("frame.jpg", img_np)
            if success:
                _LOGGER.info("SAVED FRAME")
            for id, corner in zip(ids[0], corners[0], strict=True):
                if id == self.id:
                    return corner[0] / [img_np.shape[1], img_np.shape[0]]
        else:
            _LOGGER.info("No detection")

            success = cv2.imwrite("frame.jpg", img_np)
            if success:
                _LOGGER.info("SAVED FRAME")

        return None


class GarageDoorPlanner:
    THRESHOLD = 0.01
    TIMEOUT = 30
    MIN_TIME = 5

    def __init__(self):
        self.state = PlannerState.UNKNOWN
        self.inited = False
        self.close_pos = None
        self.button_press_time = None

    def setup(self, pos):
        self.inited = True
        self.close_pos = pos
        self.state = PlannerState.CLOSE

    def open_cover(self):
        self.handle_button_pressed()
        self.state = PlannerState.OPENING

    def close_cover(self):
        self.handle_button_pressed()
        self.state = PlannerState.CLOSING

    def handle_button_pressed(self):
        self.button_press_time = datetime.now()

    def _timed_out(self) -> bool:
        # Timeout logic. If it takes too long, just assume it completed
        if self.button_press_time is None:
            _LOGGER.warning("Garage is opening/closing without someone triggering it")
        else:
            elapsed_sec = abs(datetime.now() - self.button_press_time).total_seconds()
            if elapsed_sec > self.TIMEOUT:
                _LOGGER.info("Force garage state due to timeout")
                self.button_press_time = None
                return True
        return False

    def _passed_min_time(self) -> bool:
        if self.button_press_time is None:
            _LOGGER.warning("Garage is opening/closing without someone triggering it")
        else:
            elapsed_sec = abs(datetime.now() - self.button_press_time).total_seconds()
            if elapsed_sec > self.MIN_TIME:
                return True
        return False

    def run(self, pos):
        if not self.inited:
            return

        dist = None

        if pos is not None:
            dist = np.linalg.norm(pos - self.close_pos)
            _LOGGER.info(f"Dist from close: {dist}")

        match self.state:
            case PlannerState.UNKNOWN:
                # Only way we can get here is if we're not inited
                pass
            case PlannerState.OPEN:
                if dist is None:
                    # 1. The gate is moving due to remote
                    # 2. Bad detection
                    # Do nothing
                    pass
                if dist is not None:
                    if dist < self.THRESHOLD:
                        # 1. Finished closing, so say its closed
                        self.state = PlannerState.CLOSE
            case PlannerState.CLOSE:
                if dist is None:
                    # 1. The gate is moving due to remote
                    # 2. Bad detection
                    # Do nothing
                    pass
                if dist is not None:
                    if dist > self.THRESHOLD:
                        # 1. Finished opening, so say its open
                        # 2. TODO: in the process of opening, no way of knowing rn
                        self.state = PlannerState.OPEN
            case PlannerState.OPENING:
                if dist is None:
                    # 1. The gate is moving due to remote
                    # 2. Bad detection
                    force_state = self._timed_out()
                    if force_state:
                        self.state = PlannerState.OPEN

                if dist is not None:
                    if dist > self.THRESHOLD and self._passed_min_time:
                        # 1. Finished opening, so say its open
                        self.state = PlannerState.OPEN

            case PlannerState.CLOSING:
                if dist is None:
                    # 1. The gate is moving due to remote
                    # 2. Bad detection
                    # Do nothing
                    force_state = self._timed_out()
                    if force_state:
                        # TODO: Assuming its closed seems sus
                        self.state = PlannerState.CLOSE
                if dist is not None:
                    if dist < self.THRESHOLD and self._passed_min_time:
                        # 1. Finished closing, so say its closed
                        self.state = PlannerState.CLOSE


class GarageDoorVisionCoordinator(DataUpdateCoordinator):
    def __init__(self, hass: HomeAssistant, entry, update_interval):
        super().__init__(
            hass=hass,
            logger=_LOGGER,
            name="Garage Door Vision",
            update_interval=update_interval,
        )

        self.store = Store(
            hass, STORAGE_VERSION, f"{STORAGE_KEY}_{entry.data[CONF_CAMERA]}"
        )

        self.camera_entity = entry.data[CONF_CAMERA]
        self.garage_door = GarageDoorTracker()
        self.garage_planner = GarageDoorPlanner()

    async def _async_update_data(self):
        _LOGGER.info("HELLO WORLD")

        frame = await get_camera_frame(self.hass, self.camera_entity)
        pos = self.garage_door.pos(frame)
        self.garage_planner.run(pos)

        if pos is not None:
            _LOGGER.info(f"Garage position: {pos}")

    async def async_calibrate(self):
        _LOGGER.info("Calibrating")
        frame = await get_camera_frame(self.hass, self.camera_entity)
        pos = self.garage_door.pos(frame)

        if pos is None:
            _LOGGER.error("Calibration failed: no frame")
            return False

        calibration_data = {"close_position": pos.tolist()}
        await self.store.async_save(calibration_data)

        self.garage_planner.setup(calibration_data["close_position"])
        _LOGGER.info(f"Garage position: {pos}")

        return True

    async def async_load_calibration(self):
        calibration_data = await self.store.async_load()
        if calibration_data:
            _LOGGER.info("Loaded calibration from storage")
            self.garage_planner.setup(np.array(calibration_data["close_position"]))
        else:
            _LOGGER.info("No existing calibration found")
