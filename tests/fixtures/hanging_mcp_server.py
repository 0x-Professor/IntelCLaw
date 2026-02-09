import time


def main() -> None:
    # Intentionally never speaks MCP over stdio.
    # Used to validate that MCP client startup/list_tools timeouts work.
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
