"""Appearance preferences survive restarts and reject invalid updates atomically."""
import json
from PySide6.QtCore import QSettings
from test_bridge import controller_with_data


def appearance_controller(tmp_path):
    app, controller, bridge = controller_with_data(tmp_path / "workspace")
    settings_path = tmp_path / "appearance.ini"
    controller.settings = QSettings(str(settings_path), QSettings.Format.IniFormat)
    return app, controller, bridge, settings_path


def test_dark_theme_is_persisted_and_exposed_to_renderer(tmp_path):
    app, controller, bridge, settings_path = appearance_controller(tmp_path)
    try:
        bridge.command("preferences", json.dumps({"theme": "dark"}))
        assert bridge.snapshot()["preferences"]["theme"] == "dark"
        controller.settings.sync()
        reopened = QSettings(str(settings_path), QSettings.Format.IniFormat)
        assert reopened.value("referenceTheme", "", type=str) == "dark"
        assert bridge.snapshot()["preferences"]["reduce_motion"] is False
    finally:
        controller.close()


def test_existing_light_theme_preferences_remain_supported(tmp_path):
    app, controller, bridge, settings_path = appearance_controller(tmp_path)
    try:
        for theme in ("contrast", "pastel"):
            bridge.command("preferences", json.dumps({"theme": theme}))
            assert bridge.snapshot()["preferences"]["theme"] == theme
    finally:
        controller.close()


def test_reduced_motion_preference_persists_and_can_be_disabled(tmp_path):
    app, controller, bridge, settings_path = appearance_controller(tmp_path)
    try:
        bridge.command("preferences", json.dumps({"reduce_motion": True}))
        assert bridge.snapshot()["preferences"]["reduce_motion"] is True
        controller.settings.sync()
        reopened = QSettings(str(settings_path), QSettings.Format.IniFormat)
        assert reopened.value("referenceReduceMotion", False, type=bool) is True
        bridge.command("preferences", json.dumps({"reduce_motion": False}))
        assert bridge.snapshot()["preferences"]["reduce_motion"] is False
        controller.settings.sync()
        reopened.sync()
        assert reopened.value("referenceReduceMotion", True, type=bool) is False
    finally:
        controller.close()


def test_unsupported_theme_does_not_partially_save_other_preferences(tmp_path):
    app, controller, bridge, settings_path = appearance_controller(tmp_path)
    notices = []
    bridge.notice.connect(lambda payload: notices.append(json.loads(payload)))
    try:
        bridge.command("preferences", json.dumps({
            "theme": "pastel", "reduce_motion": False,
            "research_rows": 20000, "autosave": False,
        }))
        before = {key: controller.settings.value(key) for key in (
            "referenceTheme", "referenceReduceMotion",
            "researchRowBudget", "referenceAutosave",
        )}
        bridge.command("preferences", json.dumps({
            "theme": "unsupported-theme", "reduce_motion": True,
            "research_rows": 1000, "autosave": True,
        }))
        assert notices and notices[-1]["kind"] == "error"
        after = {key: controller.settings.value(key) for key in before}
        assert after == before
    finally:
        controller.close()


def test_appearance_can_change_during_training_but_resource_mutations_stay_locked(tmp_path):
    app, controller, bridge, _ = appearance_controller(tmp_path)
    notices=[]
    bridge.notice.connect(lambda payload:notices.append(json.loads(payload)))
    try:
        controller.busy=True
        bridge.command("preferences",json.dumps({"theme":"dark","reduce_motion":True}))
        assert bridge.snapshot()["preferences"]["theme"]=="dark"
        assert bridge.snapshot()["preferences"]["reduce_motion"] is True
        bridge.command("preferences",json.dumps({"research_rows":1000}))
        assert notices[-1]["kind"]=="error"
        assert bridge.snapshot()["preferences"]["research_rows"]==20000
    finally:
        controller.busy=False
        controller.close()


def test_existing_matplotlib_chart_recolors_without_changing_data(tmp_path):
    import base64
    app, controller, bridge, _ = appearance_controller(tmp_path)
    try:
        bridge.command("chart",json.dumps({"type":"Box plot","x":"x","y":"x"}))
        lines=[line.get_xydata().copy() for line in controller.figure.axes[0].lines]
        bridge.command("preferences",json.dumps({"theme":"dark"}))
        assert "#1a2133" in base64.b64decode(bridge.chart.split(",",1)[1]).decode()
        for before,after in zip(lines,controller.figure.axes[0].lines):
            import numpy as np
            np.testing.assert_array_equal(before,after.get_xydata())
        bridge.command("preferences",json.dumps({"theme":"pastel"}))
        assert "#ffffff" in base64.b64decode(bridge.chart.split(",",1)[1]).decode()
    finally:
        controller.close()
