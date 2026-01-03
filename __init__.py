from __future__ import annotations

from datetime import timedelta

from homeassistant.core import HomeAssistant

from .const import DOMAIN
from .coordinator import GarageDoorVisionCoordinator


async def async_setup_entry(hass: HomeAssistant, entry):
    coordinator = GarageDoorVisionCoordinator(
        hass,
        entry,
        update_interval=timedelta(seconds=1),
    )

    await coordinator.async_load_calibration()
    await coordinator.async_refresh()

    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = coordinator

    await hass.config_entries.async_forward_entry_setups(
        entry,
        ["cover"],
    )

    return True


async def async_unload_entry(hass, entry):
    hass.data[DOMAIN].pop(entry.entry_id)
    return True
