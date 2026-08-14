"""Import every plugin module inside a real QGIS installation."""

import importlib
import sys
from pathlib import Path

from qgis.core import Qgis, QgsApplication


SUPPORTED_QGIS_MAJOR_VERSIONS = {3, 4}
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
    qgis_major_version = Qgis.QGIS_VERSION_INT // 10000
    if qgis_major_version not in SUPPORTED_QGIS_MAJOR_VERSIONS:
        raise RuntimeError(
            f"Expected QGIS major version 3 or 4, found {Qgis.QGIS_VERSION} "
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
