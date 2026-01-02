from __future__ import annotations

from homeassistant.components.camera import async_get_image
from homeassistant.core import HomeAssistant


async def get_camera_frame(hass: HomeAssistant, camera_entity_id: str):
    try:
        image = await async_get_image(
            hass,
            camera_entity_id,
            timeout=10,
        )

        if image is None:
            return None

        return image.content  # bytes (JPEG/PNG)

    except Exception:
        return None
