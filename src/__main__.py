from utils import PDDataLoader

def main() -> None:
    """
    Main function.
    """
    logger.info(
        f"Current module: {get_module_name()}, tzname: {datetime.now().astimezone().tzname()}, localzone: {get_localzone()}"
    )


if __name__ == "__main__":
    main()
