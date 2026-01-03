from __future__ import annotations

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.helpers.selector import (
    selector,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from .const import CONF_BUTTON, CONF_CAMERA, DOMAIN, CONF_TRACKER_URL


class GarageDoorVisionConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    VERSION = 1

    async def async_step_user(self, user_input=None):
        if user_input is not None:
            return self.async_create_entry(
                title="Garage Door Vision",
                data=user_input,
            )

        schema = vol.Schema(
            {
                vol.Required(CONF_CAMERA): selector({"entity": {"domain": "camera"}}),
                vol.Required(CONF_BUTTON): selector({"entity": {"domain": "button"}}),
                vol.Required(CONF_TRACKER_URL): TextSelector(
                    TextSelectorConfig(type=TextSelectorType.URL)
                ),
            }
        )

        return self.async_show_form(
            step_id="user",
            data_schema=schema,
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        return GarageDoorVisionOptionsFlow(config_entry)


class GarageDoorVisionOptionsFlow(config_entries.OptionsFlow):
    def __init__(self, config_entry):
        pass
        # self.config_entry = config_entry

    async def async_step_init(self, user_input=None):
        return self.async_show_menu(
            step_id="init", menu_options={"calibrate": "Calibrate"}
        )

    async def async_step_calibrate(self, user_input=None):
        if user_input is None:
            return self.async_show_form(
                step_id="calibrate",
                data_schema=vol.Schema({}),
            )

        # Trigger calibration here
        coordinator = self.hass.data[DOMAIN][self.config_entry.entry_id]
        success = await coordinator.async_calibrate()

        if success:
            return self.async_create_entry(title="", data={})

        return self.async_show_form(
            step_id="calibrate",
            data_schema=vol.Schema({}),
        )
