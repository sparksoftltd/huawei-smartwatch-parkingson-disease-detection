from loguru import logger
from python_boilerplate.common.common_function import get_module_name
from datetime import datetime

def main() -> None:
    """
    Main function.
    """
    logger.info(
        f"Current module: {get_module_name()}, tzname: {datetime.now().astimezone().tzname()}, localzone: {get_localzone()}"
    )


if __name__ == "__main__":
    main()
