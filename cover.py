from __future__ import annotations

from homeassistant.components.cover import (
    CoverDeviceClass,
    CoverEntity,
    CoverEntityFeature,
)
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN, PlannerState


async def async_setup_entry(hass, entry, async_add_entities):
    coordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([GarageDoorCover(coordinator, entry)])


class GarageDoorCover(CoordinatorEntity, CoverEntity):
    _attr_device_class = CoverDeviceClass.GATE
    _attr_supported_features = CoverEntityFeature.OPEN | CoverEntityFeature.CLOSE
    _attr_has_entity_name = True
    _attr_name = "Garage Door"

    def __init__(self, coordinator, entry):
        super().__init__(coordinator)
        self._button_entity = entry.data["button_entity"]

    @property
    def is_closed(self):
        return self.coordinator.garage_planner.state == PlannerState.CLOSE

    @property
    def is_opening(self):
        return self.coordinator.garage_planner.state == PlannerState.OPENING

    @property
    def is_closing(self):
        return self.coordinator.garage_planner.state == PlannerState.CLOSING

    async def _press_button(self):
        await self.hass.services.async_call(
            domain="button",
            service="press",
            service_data={"entity_id": self._button_entity},
            blocking=True,
        )

    async def async_open_cover(self, **kwargs):
        await self._press_button()
        self.coordinator.garage_planner.open_cover()

    async def async_close_cover(self, **kwargs):
        await self._press_button()
        self.coordinator.garage_planner.close_cover()
