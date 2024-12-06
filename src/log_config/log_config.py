""" "Configure Logging"""

import logging


def configure_logging(debug: bool = False):
    """
    Configure logging

    Args:
        debug (bool): Enable debugging

    Returns:
        None
    """
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("app.log")],
    )
