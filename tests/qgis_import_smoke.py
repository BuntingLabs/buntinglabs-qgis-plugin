"""Import every plugin module inside a real QGIS installation."""

import importlib
import sys
from pathlib import Path

from qgis.core import Qgis, QgsApplication


EXPECTED_QGIS_VERSION = 34412
PLUGIN_DIRECTORY = Path(__file__).resolve().parents[1]
PLUGIN_PACKAGE = PLUGIN_DIRECTORY.name
PLUGIN_MODULES = (
    "ai_tracer",
    "digitize_ld_task",
    "login_check_task",
    "onboarding_widget",
    "plugin",
    "tracing_task",
    "trajectory_tree",
)


def main() -> None:
    if Qgis.QGIS_VERSION_INT != EXPECTED_QGIS_VERSION:
        raise RuntimeError(
            f"Expected QGIS 3.44.12, found {Qgis.QGIS_VERSION} "
            f"({Qgis.QGIS_VERSION_INT})"
        )

    sys.path.insert(0, str(PLUGIN_DIRECTORY.parent))

    QgsApplication.setPrefixPath("/usr", True)
    application = QgsApplication([], False)
    application.initQgis()

    try:
        package = importlib.import_module(PLUGIN_PACKAGE)
        if not callable(getattr(package, "classFactory", None)):
            raise RuntimeError("Plugin package does not expose classFactory")

        for module_name in PLUGIN_MODULES:
            importlib.import_module(f"{PLUGIN_PACKAGE}.{module_name}")
    finally:
        application.exitQgis()

    print(f"Imported {PLUGIN_PACKAGE} successfully with QGIS {Qgis.QGIS_VERSION}")


if __name__ == "__main__":
    main()
