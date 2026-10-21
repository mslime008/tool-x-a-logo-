"""Only non-secret desktop preferences are persisted."""
from PySide6.QtCore import QSettings


def preferences() -> QSettings:
    return QSettings("LogoRemover", "VideoLogoRemover")
